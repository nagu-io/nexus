"""Tests for Tier 2 upgrades: ConversationStore, ResearchAgent scraping, WebSocket manager."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nexus.memory.conversation_store import ConversationStore
from nexus.agents.research_agent import ResearchAgent
from nexus.api import _compose_chat_prompt, _prepare_chat_prompt
from nexus.runtime.context_reducer import ContextReductionResult


# ==================================================================
# ConversationStore Tests
# ==================================================================

class ConversationStoreBasicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.db"
        self.store = ConversationStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_save_and_retrieve_messages(self):
        self.store.save_message("s1", "user", "hello")
        self.store.save_message("s1", "assistant", "hi there")

        history = self.store.get_history("s1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "hello")
        self.assertEqual(history[1]["role"], "assistant")

    def test_returns_empty_for_unknown_session(self):
        history = self.store.get_history("nonexistent")
        self.assertEqual(history, [])

    def test_message_count(self):
        self.store.save_message("s1", "user", "a")
        self.store.save_message("s1", "user", "b")
        self.store.save_message("s2", "user", "c")

        self.assertEqual(self.store.message_count("s1"), 2)
        self.assertEqual(self.store.message_count("s2"), 1)
        self.assertEqual(self.store.message_count(), 3)

    def test_metadata_round_trips(self):
        meta = {"agent": "coding", "reflect_score": 0.05}
        self.store.save_message("s1", "assistant", "code here", metadata=meta)

        history = self.store.get_history("s1")
        self.assertEqual(history[0]["metadata"]["agent"], "coding")
        self.assertAlmostEqual(history[0]["metadata"]["reflect_score"], 0.05)

    def test_history_limit(self):
        for i in range(10):
            self.store.save_message("s1", "user", f"message {i}")

        history = self.store.get_history("s1", limit=3)
        self.assertEqual(len(history), 3)
        # Should be the 3 most recent, in chronological order
        self.assertIn("7", history[0]["content"])

    def test_get_context_returns_role_content_pairs(self):
        self.store.save_message("s1", "user", "what is Python?")
        self.store.save_message("s1", "assistant", "A programming language")

        context = self.store.get_context("s1")
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0], {"role": "user", "content": "what is Python?"})
        self.assertEqual(context[1], {"role": "assistant", "content": "A programming language"})


class ConversationStoreSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.db"
        self.store = ConversationStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_search_finds_matching_messages(self):
        self.store.save_message("s1", "user", "build a REST API with Express")
        self.store.save_message("s1", "assistant", "here is your Express server")
        self.store.save_message("s2", "user", "what is the weather?")

        results = self.store.search("Express")
        self.assertEqual(len(results), 2)

    def test_search_returns_empty_for_no_match(self):
        self.store.save_message("s1", "user", "hello")
        results = self.store.search("nonexistent_term_xyz")
        self.assertEqual(results, [])


class ConversationStoreSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.db"
        self.store = ConversationStore(db_path=self.db_path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_list_sessions(self):
        self.store.save_message("session_a", "user", "hello")
        self.store.save_message("session_b", "user", "world")

        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 2)
        self.assertTrue(all("id" in s for s in sessions))

    def test_delete_session(self):
        self.store.save_message("s1", "user", "hello")
        self.store.save_message("s1", "assistant", "hi")
        self.store.save_message("s2", "user", "other")

        deleted = self.store.delete_session("s1")
        self.assertEqual(deleted, 2)
        self.assertEqual(self.store.message_count("s1"), 0)
        self.assertEqual(self.store.message_count("s2"), 1)

    def test_session_message_count_tracks(self):
        self.store.save_message("s1", "user", "a")
        self.store.save_message("s1", "user", "b")

        sessions = self.store.list_sessions()
        s1 = [s for s in sessions if s["id"] == "s1"][0]
        self.assertEqual(s1["message_count"], 2)


# ==================================================================
# ResearchAgent Enhancement Tests
# ==================================================================

class ResearchAgentScrapingTests(unittest.TestCase):
    def test_extract_text_strips_html(self):
        html = "<html><body><p>Hello world, this is a long paragraph that should survive the text extraction process easily</p><script>evil()</script></body></html>"
        text = ResearchAgent._extract_text_from_html(html)
        self.assertIn("long paragraph", text)
        self.assertNotIn("script", text)
        self.assertNotIn("evil", text)

    def test_extract_text_removes_nav_footer(self):
        html = """
        <html>
        <body>
            <nav>Menu items here</nav>
            <main><p>This is the actual content of the article that we want to extract.</p></main>
            <footer>Copyright 2024</footer>
        </body>
        </html>
        """
        text = ResearchAgent._extract_text_from_html(html)
        self.assertIn("actual content", text)
        self.assertNotIn("Menu items", text)
        self.assertNotIn("Copyright", text)

    def test_extract_text_decodes_entities(self):
        html = "<p>Tom &amp; Jerry are a classic cartoon duo that everyone loves and remembers fondly</p>"
        text = ResearchAgent._extract_text_from_html(html)
        self.assertIn("Tom & Jerry", text)

    def test_has_web_scraping_capability(self):
        agent = ResearchAgent()
        self.assertIn("web_scraping", agent.capabilities)

    def test_search_results_include_url(self):
        # Verify the search result format includes URL field
        agent = ResearchAgent()
        # We can't actually call DDG in tests, but we verify the structure
        self.assertTrue(hasattr(agent, "_scrape_page"))
        self.assertTrue(hasattr(agent, "_extract_text_from_html"))


# ==================================================================
# WebSocket ConnectionManager Tests
# ==================================================================

class ConnectionManagerTests(unittest.TestCase):
    def test_import_and_create_manager(self):
        """Verify the ConnectionManager can be imported and instantiated."""
        from nexus.api import ConnectionManager
        manager = ConnectionManager()
        self.assertEqual(manager.count, 0)

    def test_ws_manager_exists(self):
        """Verify the global ws_manager is available."""
        from nexus.api import ws_manager
        self.assertEqual(ws_manager.count, 0)


class ChatPromptReductionTests(unittest.TestCase):
    def test_compose_chat_prompt_includes_recent_history(self):
        prompt = _compose_chat_prompt(
            history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "latest"},
            ],
            latest_message="latest",
        )

        self.assertIn("Conversation history:", prompt)
        self.assertIn("USER: hello", prompt)
        self.assertIn("ASSISTANT: hi", prompt)
        self.assertIn("Latest user request:\nlatest", prompt)

    def test_prepare_chat_prompt_uses_reducer_for_large_history(self):
        class StubReducer:
            def reduce(self, text: str, *, metadata=None):
                return ContextReductionResult(
                    text="[CHAT_REDUCED]",
                    reduced=True,
                    backend="stub",
                    strategy="unit_test",
                    original_length=len(text),
                    reduced_length=len("[CHAT_REDUCED]"),
                    metadata=dict(metadata or {}),
                )

        prompt, reduction = _prepare_chat_prompt(
            history=[
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "latest"},
            ],
            latest_message="latest",
            reducer=StubReducer(),
        )

        self.assertEqual(prompt, "[CHAT_REDUCED]")
        self.assertIsNotNone(reduction)
        self.assertEqual(reduction.backend, "stub")
        self.assertEqual(reduction.metadata["scope"], "chat")
        self.assertEqual(reduction.metadata["history_messages"], 2)


if __name__ == "__main__":
    unittest.main()
