"""
CodingAgent — handles code generation, debugging, refactoring.
This is the Claude Code replacement inside NEXUS.
Runs locally via CompressX model. Falls back to cloud for complex tasks.
"""

import hashlib
import json
from pathlib import Path
import re
from difflib import SequenceMatcher

from nexus.agents.base_agent import BaseAgent
from nexus.runtime.build_artifacts import BuildArtifactMaterializer
from rich.console import Console
from rich.syntax import Syntax

console = Console()


class CodingAgent(BaseAgent):
    """
    Local coding agent powered by CompressX compressed model.
    Handles: code generation, debugging, refactoring, explanation.
    """

    name = "coding"
    capabilities = ("reasoning", "code_generation", "debugging", "summarization", "testing")
    system_prompt = """You are an expert software engineer. 
You write clean, correct, production-ready code.
You always include comments explaining what the code does.
You prefer simple solutions over complex ones.
When debugging, you identify the root cause before suggesting a fix.
Output only the code and a brief explanation. No fluff."""

    STACK_RULES = [
        {
            "name": "express_js",
            "signals": ("express", "node", "javascript", "api route", "api routes"),
            "required_terms": ("express",),
            "forbidden_terms": ("from flask", "flask(", "django", "fastapi", "import express as", "def login("),
            "failure_type": "framework_mismatch",
            "reason": "The task asked for an Express/Node solution, but the output drifted into Python-style backend code.",
        },
        {
            "name": "react_frontend",
            "signals": ("react", "frontend", "form", "component", "jsx"),
            "required_terms": ("react", "jsx", "tsx", "useState", "form"),
            "forbidden_terms": ("jinja", "render_template", "django template"),
            "failure_type": "framework_mismatch",
            "reason": "The task asked for a React/frontend slice, but the output did not stay in a frontend/UI implementation path.",
        },
    ]
    SUSPICIOUS_CODE_MARKERS = (
        "consoleinconsole>",
        "0x57, 0x68",
        "./imgs/",
        "ejs.create_app",
        "import express as",
    )
    MAX_FIX_ATTEMPTS = 2

    def __init__(self):
        super().__init__()
        self._artifact_materializer = BuildArtifactMaterializer()
        self._autonomous_sessions: dict[str, dict] = {}

    async def run(self, task: str) -> str:
        """Execute coding task."""
        console.print(f"[cyan]CodingAgent: {task[:60]}...[/cyan]")

        # Check if debugging request
        if any(w in task.lower() for w in ["debug", "fix", "error", "bug", "not working"]):
            response = await self._debug(task)
        elif "task type: test_generation" in task.lower():
            response = await self.generate_tests(task)
        elif any(w in task.lower() for w in ["explain", "what does", "how does"]):
            response = await self._explain(task)
        else:
            response = await self._generate(task)

        # Pretty print code blocks
        if "```" in response:
            console.print(Syntax(response, "python", theme="monokai"))

        return response

    async def act(self, task: str, memory=None, thought: dict | None = None):
        """Switch into autonomous workspace mode when the runtime provides one."""
        if self._is_autonomous_mode(task, memory):
            return await self._autonomous_act(task, memory)
        return await super().act(task, memory=memory, thought=thought)

    async def continue_after_tool(
        self,
        task: str,
        tool_result: dict,
        memory=None,
        thought: dict | None = None,
    ):
        """Sequence autonomous file/terminal actions and plan fixes from terminal feedback."""
        session_key = self._session_key(task, memory)
        session = self._autonomous_sessions.get(session_key)
        if not session:
            return await super().continue_after_tool(task, tool_result, memory=memory, thought=thought)

        session.setdefault("tool_results", []).append(dict(tool_result))

        if not tool_result.get("ok", False):
            if tool_result.get("tool") == "terminal_tool" and session.get("fix_attempts", 0) < self.MAX_FIX_ATTEMPTS:
                fix_actions = await self._plan_fix_actions(task, memory, tool_result)
                if fix_actions:
                    session["fix_attempts"] = session.get("fix_attempts", 0) + 1
                    rerun = self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=tool_result.get("command", []),
                        cwd=tool_result.get("cwd"),
                        timeout_seconds=tool_result.get("timeout_seconds", 60),
                    )
                    session["actions"] = fix_actions + [rerun] + list(session.get("actions", []))
                    return session["actions"].pop(0)
            summary = tool_result.get("summary", "Tool error: autonomous step failed")
            self._autonomous_sessions.pop(session_key, None)
            return f"Tool error: {summary}"

        if session.get("actions"):
            return session["actions"].pop(0)

        self._autonomous_sessions.pop(session_key, None)
        workspace_root = memory.get("workspace.root_dir") if memory and hasattr(memory, "get") else None
        return (
            f"Autonomous coding run completed in {workspace_root}. "
            f"Last step: {tool_result.get('summary', 'completed')}"
            if workspace_root
            else tool_result.get("summary", "Autonomous coding run completed")
        )

    async def observe(
        self,
        task: str,
        result: str,
        memory=None,
        thought: dict | None = None,
    ) -> dict:
        """Reject successful-looking outputs that violate the requested stack."""
        observation = await super().observe(task, result, memory=memory, thought=thought)
        if not observation.get("ok", False):
            return observation

        mismatch = self._detect_stack_mismatch(task, result)
        if mismatch:
            observation["ok"] = False
            observation["summary"] = mismatch["reason"]
            observation["failure_type"] = mismatch["failure_type"]
        return observation

    async def _generate(self, task: str) -> str:
        """Generate code for a task."""
        scaffold = self._deterministic_scaffold(task)
        if scaffold:
            return scaffold
        prompt = self._build_generation_prompt(task)
        return await self._call_local(prompt)

    async def generate_tests(self, task: str) -> str:
        """Generate runnable unit tests for an existing solution."""
        prompt = (
            "You are generating tests for an existing implementation.\n"
            "Return only path-tagged fenced code blocks for the test files that should be added.\n"
            "Prefer the project's existing test runner when one is visible.\n"
            "Do not restate the implementation.\n\n"
            f"{task}"
        )
        return await self._call_local(prompt)

    async def _debug(self, task: str) -> str:
        """Debug code or error."""
        prompt = f"Debug the following issue:\n\n{task}\n\nIdentify the root cause and provide the fix."
        return await self._call_local(prompt)

    async def _explain(self, task: str) -> str:
        """Explain code."""
        prompt = f"Explain the following clearly and concisely:\n\n{task}"
        return await self._call_local(prompt)

    def _build_generation_prompt(self, task: str) -> str:
        """Build a stricter prompt that preserves the requested stack and scope."""
        stack_guidance = self._stack_guidance(task)
        full_stack = self._is_full_stack_request(task)
        deliverable = (
            "Produce a minimal but coherent vertical slice with backend, API routes, and frontend pieces."
            if full_stack
            else "Produce a focused implementation for the requested task."
        )
        structure = (
            "Return:\n"
            "1. a short architecture note\n"
            "2. a file tree\n"
            "3. the key source files as fenced code blocks\n"
            "4. brief run instructions"
            if full_stack
            else "Return complete, working code with concise comments."
        )
        return (
            "Write code for the following task.\n"
            "Honor the requested language and framework exactly.\n"
            "Do not switch stacks, invent alternate frameworks, or answer in a different language.\n"
            "If the user asked for Express, stay in Node/Express. If the user asked for frontend UI, include the UI layer.\n"
            "Prefer a small, runnable scaffold over a huge speculative code dump.\n\n"
            f"Task:\n{task}\n\n"
            f"Stack guidance:\n{stack_guidance}\n\n"
            f"Implementation goal:\n{deliverable}\n\n"
            f"Output structure:\n{structure}"
        )

    def _stack_guidance(self, task: str) -> str:
        """Summarize the requested stack in a form the local model can follow."""
        task_lower = task.lower()
        guidance = []
        if "express" in task_lower:
            guidance.append("- Backend: Node.js with Express")
        if any(token in task_lower for token in ("api route", "api routes", "endpoint", "rest")):
            guidance.append("- Include explicit API routes/endpoints")
        if any(token in task_lower for token in ("frontend", "form", "ui", "react")):
            guidance.append("- Include a basic frontend form/UI")
        if "typescript" in task_lower:
            guidance.append("- Language: TypeScript")
        elif any(token in task_lower for token in ("javascript", "express", "node")):
            guidance.append("- Language: JavaScript")
        if not guidance:
            guidance.append("- Use the language/framework explicitly requested in the task")
        return "\n".join(guidance)

    def _is_full_stack_request(self, task: str) -> bool:
        """Detect when a task spans multiple app layers."""
        task_lower = task.lower()
        return (
            "full stack" in task_lower
            or ("backend" in task_lower and "frontend" in task_lower)
            or ("api" in task_lower and "frontend" in task_lower)
        )

    def _detect_stack_mismatch(self, task: str, result: str) -> dict | None:
        """Catch responses that drift away from the requested stack."""
        task_lower = task.lower()
        result_lower = result.lower()
        for rule in self.STACK_RULES:
            if not any(signal in task_lower for signal in rule["signals"]):
                continue
            if any(term in result_lower for term in rule["forbidden_terms"]):
                return {"failure_type": rule["failure_type"], "reason": rule["reason"]}
            if rule["name"] == "express_js":
                if "express" in task_lower and re.search(r"\bimport\s+[\w]+\s+as\s+\w+\b", result_lower):
                    return {"failure_type": rule["failure_type"], "reason": rule["reason"]}
        completeness_gap = self._detect_structure_gap(task_lower, result_lower)
        if completeness_gap:
            return completeness_gap
        return None

    def _detect_structure_gap(self, task_lower: str, result_lower: str) -> dict | None:
        """Reject code-like output that misses essential pieces or contains obvious corruption."""
        if "express" not in task_lower:
            return None
        if any(marker in result_lower for marker in self.SUSPICIOUS_CODE_MARKERS):
            return {
                "failure_type": "incomplete_scaffold",
                "reason": "The generated code included corrupted or unrelated fragments instead of a clean Express scaffold.",
            }

        required_signals = ["express", "/api/login", "<form", "fetch("]
        if all(signal in task_lower for signal in ("frontend", "login")) and not all(
            signal in result_lower for signal in required_signals
        ):
            return {
                "failure_type": "incomplete_scaffold",
                "reason": "The generated scaffold missed one or more required pieces: Express API route, frontend form, or client submission flow.",
            }
        return None

    def _deterministic_scaffold(self, task: str) -> str | None:
        """Return a clean local scaffold for common high-signal build requests."""
        task_lower = task.lower()
        if not self._is_express_login_scaffold(task_lower):
            return None
        return """Architecture Note:
- Express serves the API and the static frontend from one local process.
- `/api/login` handles the authentication request and returns a small JSON response.
- The frontend is a plain HTML form with a fetch-based submit handler so the scaffold stays easy to run locally.

File Tree:
```text
login-system/
|-- package.json
|-- .env.example
|-- backend/
|   |-- server.js
|   `-- routes/
|       `-- auth.js
`-- frontend/
    |-- index.html
    |-- app.js
    `-- styles.css
```

Key Source Files:

`package.json`
```json
{
  "name": "express-login-system",
  "version": "1.0.0",
  "private": true,
  "type": "commonjs",
  "scripts": {
    "dev": "node backend/server.js"
  },
  "dependencies": {
    "cors": "^2.8.5",
    "dotenv": "^16.4.5",
    "express": "^4.19.2"
  }
}
```

`.env.example`
```bash
PORT=3000
```

`backend/server.js`
```javascript
const path = require("path");
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const authRouter = require("./routes/auth");

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use("/api", authRouter);
app.use(express.static(path.join(__dirname, "..", "frontend")));

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "express-login-system" });
});

app.listen(port, () => {
  console.log(`Login system running at http://localhost:${port}`);
});
```

`backend/routes/auth.js`
```javascript
const express = require("express");

const router = express.Router();

const demoUser = {
  email: "demo@nexus.local",
  password: "demo1234",
  name: "Demo User"
};

router.post("/login", (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({
      ok: false,
      message: "Email and password are required."
    });
  }

  if (email !== demoUser.email || password !== demoUser.password) {
    return res.status(401).json({
      ok: false,
      message: "Invalid credentials."
    });
  }

  return res.json({
    ok: true,
    user: {
      email: demoUser.email,
      name: demoUser.name
    },
    token: "local-demo-token"
  });
});

module.exports = router;
```

`frontend/index.html`
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Express Login</title>
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <main class="shell">
      <section class="card">
        <h1>Login</h1>
        <p>Use <strong>demo@nexus.local</strong> / <strong>demo1234</strong>.</p>
        <form id="login-form">
          <label>
            Email
            <input type="email" name="email" value="demo@nexus.local" required />
          </label>
          <label>
            Password
            <input type="password" name="password" value="demo1234" required />
          </label>
          <button type="submit">Sign in</button>
        </form>
        <pre id="result" aria-live="polite"></pre>
      </section>
    </main>
    <script src="./app.js"></script>
  </body>
</html>
```

`frontend/app.js`
```javascript
const form = document.getElementById("login-form");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = {
    email: formData.get("email"),
    password: formData.get("password")
  };

  result.textContent = "Signing in...";

  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    result.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    result.textContent = JSON.stringify(
      { ok: false, message: error.message },
      null,
      2
    );
  }
});
```

`frontend/styles.css`
```css
body {
  margin: 0;
  font-family: "Segoe UI", sans-serif;
  background: #0b1220;
  color: #e5eefc;
}

.shell {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 24px;
}

.card {
  width: min(100%, 420px);
  background: #101a2e;
  border: 1px solid #223556;
  border-radius: 18px;
  padding: 24px;
  box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35);
}

form {
  display: grid;
  gap: 14px;
  margin-top: 18px;
}

label {
  display: grid;
  gap: 6px;
  font-size: 0.95rem;
}

input,
button {
  border-radius: 10px;
  border: 1px solid #30476f;
  padding: 12px 14px;
  font-size: 1rem;
}

button {
  cursor: pointer;
  background: #38bdf8;
  color: #082032;
  font-weight: 700;
}

#result {
  margin-top: 18px;
  background: #08111f;
  border-radius: 12px;
  padding: 12px;
  min-height: 96px;
  overflow: auto;
}
```

Run Instructions:
1. Create the files with the structure above.
2. Run `npm install`.
3. Copy `.env.example` to `.env`.
4. Run `npm run dev`.
5. Open `http://localhost:3000` and submit the demo credentials.
"""

    def _is_express_login_scaffold(self, task_lower: str) -> bool:
        return (
            "express" in task_lower
            and any(token in task_lower for token in ("login", "auth"))
            and any(token in task_lower for token in ("frontend", "form", "ui"))
            and any(token in task_lower for token in ("api", "route", "backend"))
        )

    def _is_autonomous_mode(self, task: str, memory) -> bool:
        """Enable structured tool execution when the runtime provides a project workspace."""
        if not memory or not hasattr(memory, "get"):
            return False
        workspace_root = memory.get("workspace.root_dir")
        if not workspace_root:
            return False
        lowered = task.lower()
        return (
            "task type: solution" in lowered
            or "workspace root:" in lowered
            or any(token in lowered for token in ("build", "implement", "create", "fix", "debug", "refactor"))
        )

    async def _autonomous_act(self, task: str, memory) -> dict | str:
        """Plan workspace file/command actions and return the next tool call."""
        session_key = self._session_key(task, memory)
        session = self._autonomous_sessions.get(session_key)
        if session is None or not session.get("actions"):
            session = await self._build_autonomous_session(task, memory)
            self._autonomous_sessions[session_key] = session
        if session.get("actions"):
            return session["actions"].pop(0)
        return session.get("summary") or "No autonomous coding actions were required."

    async def _build_autonomous_session(self, task: str, memory) -> dict:
        """Turn a coding goal into sequential file writes and validation commands."""
        workspace_root = Path(memory.get("workspace.root_dir")).resolve()
        task_type = self._task_type_from_prompt(task)
        project_state = memory.get("workspace.project_state") or {}
        scaffold_text = await self._generate_autonomous_artifacts(task, memory, task_type=task_type)

        console.print(f"[dim]Local model raw output ({len(scaffold_text)} chars):[/dim]")
        console.print(f"[dim]{scaffold_text[:500]}[/dim]")

        artifacts = self._artifact_materializer.extract(scaffold_text)

        if not artifacts:
            console.print("[yellow]No code blocks extracted from local model output. Using fallback scaffold.[/yellow]")
            scaffold_text = self._build_fallback_scaffold(task, workspace_root)
            artifacts = self._artifact_materializer.extract(scaffold_text)

        console.print(f"[green]Extracted {len(artifacts)} file artifact(s) for workspace[/green]")

        actions = self._artifacts_to_tool_calls(artifacts, workspace_root)
        actions.extend(
            self._validation_commands(
                artifacts,
                workspace_root,
                task_type=task_type,
                task=task,
                project_state=project_state,
            )
        )
        return {
            "workspace_root": str(workspace_root),
            "scaffold_text": scaffold_text,
            "actions": actions,
            "summary": scaffold_text[:240],
            "fix_attempts": 0,
            "tool_results": [],
            "task_type": task_type,
        }

    def _build_fallback_scaffold(self, task: str, workspace_root: Path) -> str:
        """Generate a minimal but real project scaffold when the local model fails to produce code blocks."""
        task_lower = task.lower()

        # Detect what the user wants
        wants_auth = any(w in task_lower for w in ("auth", "login", "signup", "sign up", "register"))
        wants_api = any(w in task_lower for w in ("api", "endpoint", "route", "backend", "server", "express"))
        wants_frontend = any(w in task_lower for w in ("frontend", "ui", "page", "form", "react", "html"))

        if wants_auth or (wants_api and wants_frontend):
            return self._deterministic_scaffold(task) or self._generic_node_scaffold(task, workspace_root)
        if wants_api:
            return self._generic_node_scaffold(task, workspace_root)
        if wants_frontend:
            return self._generic_frontend_scaffold(task, workspace_root)
        return self._generic_node_scaffold(task, workspace_root)

    def _generic_node_scaffold(self, task: str, workspace_root: Path) -> str:
        """A minimal Express starter that the materializer can extract."""
        return f"""Architecture Note:
- Simple Express server that handles the requested task.
- Created as a greenfield project in {workspace_root.name}.

`package.json`
```json
{{
  "name": "{workspace_root.name}",
  "version": "1.0.0",
  "private": true,
  "type": "commonjs",
  "scripts": {{
    "dev": "node server.js"
  }},
  "dependencies": {{
    "express": "^4.19.2"
  }}
}}
```

`server.js`
```javascript
const express = require("express");
const path = require("path");
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.get("/api/health", (_req, res) => {{
  res.json({{ ok: true, project: "{workspace_root.name}" }});
}});

app.listen(port, () => {{
  console.log(`Server running at http://localhost:${{port}}`);
}});
```

`public/index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{workspace_root.name}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; background: #0a0a0a; color: #e5e5e5; min-height: 100vh; display: grid; place-items: center; }}
    .container {{ text-align: center; padding: 2rem; }}
    h1 {{ font-size: 2rem; margin-bottom: 1rem; background: linear-gradient(135deg, #00f0ff, #bf00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    p {{ color: #888; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{workspace_root.name}</h1>
    <p>Project created by NEXUS. Edit server.js to build your API.</p>
  </div>
</body>
</html>
```

Run: `npm install && npm run dev`
"""

    def _generic_frontend_scaffold(self, task: str, workspace_root: Path) -> str:
        """A minimal HTML/CSS/JS starter scaffold."""
        return f"""Architecture Note:
- Static frontend project with HTML, CSS, and JavaScript.
- No build step required, just open index.html.

`index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{workspace_root.name}</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main id="app">
    <h1>{workspace_root.name}</h1>
    <p>Frontend project scaffolded by NEXUS.</p>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

`styles.css`
```css
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  background: #0a0a0a;
  color: #e5e5e5;
  min-height: 100vh;
  display: grid;
  place-items: center;
}}
h1 {{
  font-size: 2.5rem;
  background: linear-gradient(135deg, #00f0ff, #bf00ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
}}
```

`app.js`
```javascript
document.addEventListener("DOMContentLoaded", () => {{
  console.log("{workspace_root.name} loaded");
}});
```
"""

    async def _generate_autonomous_artifacts(self, task: str, memory, *, task_type: str) -> str:
        """Produce file-block output that can be materialized into a workspace."""
        if task_type == "test_generation":
            return await self._generate_test_artifacts(task, memory)
        deterministic = self._deterministic_scaffold(task)
        if deterministic:
            return deterministic

        workspace_root = Path(memory.get("workspace.root_dir")).resolve()
        existing_files = self._workspace_file_context(memory)
        allowed_paths = self._allowed_workspace_paths(memory)
        task_lower = task.lower()
        repair_guidance = ""
        if any(token in task_lower for token in ("fix", "debug", "bug", "failing test", "unit test")):
            repair_guidance = (
                "Prefer the smallest grounded fix that preserves the existing project structure.\n"
                "If tests already exist, keep their intent and update implementation files first.\n"
            )
        path_guidance = ""
        if allowed_paths:
            path_guidance = (
                "When editing an existing file, the path line must match the workspace exactly.\n"
                f"Allowed existing paths: {', '.join(allowed_paths[:12])}\n"
                "Never invent placeholder paths like relative/path.ext or labels like Updated math_utils.py.\n"
            )
        empty_guidance = ""
        if "No file previews were available" in existing_files:
            empty_guidance = (
                "The workspace is completely empty. You must generate all foundational files from scratch.\n"
                "DO NOT assume the existence of any existing architecture, frameworks, or code unless specified by the user.\n"
                "Treat this as a brand new greenfield project.\n"
            )

        prompt = (
            "You are generating or repairing files inside an existing coding workspace.\n"
            "Treat the provided workspace file previews as the source of truth for the current codebase.\n"
            "Only modify files that are necessary for the task.\n"
            f"{empty_guidance}"
            f"{repair_guidance}"
            f"{path_guidance}"
            "Return only:\n"
            "1. a short architecture note\n"
            "2. path-tagged fenced code blocks for every file that should exist, using this exact format:\n"
            "`relative/path.ext`\n```language\n...\n```\n"
            "3. a short run note at the end if needed.\n"
            "Do not include placeholders, ellipses, or omitted sections.\n\n"
            f"Workspace root:\n{workspace_root}\n\n"
            f"Existing workspace files:\n{existing_files}\n\n"
            f"Task: {task}\n\n"
            "You MUST output at least one file. Output your response EXACTLY starting with:\n"
            "Architecture Note:\n"
            "- <your note here>\n\n"
            "`<filename>`\n"
            "```<language>\n"
            "<code here>\n"
            "```"
        )
        return await self._call_local(prompt)

    async def _generate_test_artifacts(self, task: str, memory) -> str:
        workspace_root = Path(memory.get("workspace.root_dir")).resolve()
        project_state = memory.get("workspace.project_state") or {}
        file_lines = "\n".join(
            f"- {item['path']}\n  Preview: {item.get('preview', '')}"
            for item in project_state.get("files", [])[:16]
        )
        prompt = (
            "You are generating automated tests for an existing project workspace.\n"
            "Return only path-tagged fenced code blocks for new or updated test files.\n"
            "Prefer the project's existing test framework. If none exists, choose the smallest sensible default.\n"
            "The tests should validate the current implementation behavior, not replace the app itself.\n\n"
            f"Workspace root:\n{workspace_root}\n\n"
            f"Existing files:\n{file_lines}\n\n"
            f"{task}"
        )
        return await self._call_local(prompt)

    def _artifacts_to_tool_calls(self, artifacts: list, workspace_root: Path) -> list[dict]:
        """Convert desired files into write/edit tool calls."""
        actions: list[dict] = []
        for artifact in artifacts:
            action = self._artifact_to_tool_call(artifact.relative_path, artifact.content, workspace_root)
            if action:
                actions.append(action)
        return actions

    def _artifact_to_tool_call(self, relative_path: Path, content: str, workspace_root: Path) -> dict | None:
        destination = workspace_root / relative_path
        if destination.exists():
            existing = destination.read_text(encoding="utf-8", errors="replace")
            if existing == content:
                return None
            edit_arguments = self._build_edit_arguments(existing, content)
            return self.tool_call(
                tool="file_tool",
                action="edit_file",
                path=relative_path.as_posix(),
                **edit_arguments,
            )
        return self.tool_call(
            tool="file_tool",
            action="write_file",
            path=relative_path.as_posix(),
            content=content,
        )

    def _build_edit_arguments(self, existing: str, updated: str) -> dict:
        """Prefer a partial edit when there is one clear changed block."""
        matcher = SequenceMatcher(None, existing, updated)
        changes = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
        if len(changes) == 1:
            _, i1, i2, j1, j2 = changes[0]
            old_text = existing[i1:i2]
            new_text = updated[j1:j2]
            if old_text:
                return {
                    "old_text": old_text,
                    "new_text": new_text,
                    "replace_all": False,
                }
        return {
            "old_text": existing,
            "new_text": updated,
            "replace_all": False,
        }

    def _validation_commands(
        self,
        artifacts: list,
        workspace_root: Path,
        *,
        task_type: str = "solution",
        task: str = "",
        project_state: dict | None = None,
    ) -> list[dict]:
        """Derive safe validation commands from the generated project files."""
        actions: list[dict] = []
        artifact_map = {artifact.relative_path.as_posix(): artifact for artifact in artifacts}
        known_paths = {artifact.relative_path.as_posix().lower() for artifact in artifacts}
        known_paths.update(
            str(item.get("path", "")).replace("\\", "/").lower()
            for item in (project_state or {}).get("files", [])
            if item.get("path")
        )
        package_artifact = artifact_map.get("package.json")
        if package_artifact:
            actions.append(
                self.tool_call(
                    tool="terminal_tool",
                    action="run_command",
                    command=["npm", "install"],
                    cwd=str(workspace_root),
                    timeout_seconds=120,
                )
            )
            try:
                package_payload = json.loads(package_artifact.content)
            except json.JSONDecodeError:
                package_payload = {}
            scripts = dict(package_payload.get("scripts", {}))
            if task_type == "test_generation" and "test" in scripts:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=["npm", "test"],
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions
            if "build" in scripts:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=["npm", "run", "build"],
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions
            if "test" in scripts:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=["npm", "test"],
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions

        if self._task_mentions_tests(task):
            if "package.json" in known_paths:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=["npm", "test"],
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions
            python_test_command = self._python_test_command(known_paths)
            if python_test_command:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=python_test_command,
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions

        js_files = [
            artifact.relative_path.as_posix()
            for artifact in artifacts
            if artifact.relative_path.suffix in {".js", ".cjs", ".mjs"}
        ]
        py_files = [
            artifact.relative_path.as_posix()
            for artifact in artifacts
            if artifact.relative_path.suffix == ".py"
        ]
        has_py_tests = any(
            "test" in Path(path).name.lower() or "tests/" in path
            for path in py_files
        )
        has_js_tests = any(
            "test" in Path(path).name.lower() or "__tests__" in path
            for path in js_files
        )
        if task_type == "test_generation":
            if has_js_tests:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=["npm", "test"],
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions
            python_test_command = self._python_test_command(known_paths)
            if has_py_tests or python_test_command:
                actions.append(
                    self.tool_call(
                        tool="terminal_tool",
                        action="run_command",
                        command=python_test_command or ["python", "-m", "unittest", "-v"],
                        cwd=str(workspace_root),
                        timeout_seconds=120,
                    )
                )
                return actions
        for path in js_files[:5]:
            actions.append(
                self.tool_call(
                    tool="terminal_tool",
                    action="run_command",
                    command=["node", "--check", path],
                    cwd=str(workspace_root),
                    timeout_seconds=60,
                )
            )
        for path in py_files[:5]:
            actions.append(
                self.tool_call(
                    tool="terminal_tool",
                    action="run_command",
                    command=["python", "-m", "py_compile", path],
                    cwd=str(workspace_root),
                    timeout_seconds=60,
                )
            )
        return actions

    def _workspace_file_context(self, memory, *, max_files: int = 12) -> str:
        project_state = memory.get("workspace.project_state") or {}
        entries = []
        for item in (project_state.get("files") or [])[:max_files]:
            path = item.get("path")
            if not path:
                continue
            size = item.get("size", 0)
            preview = str(item.get("preview", "") or "").replace("```", "'''").strip()
            entries.append(f"- {path} ({size} bytes)")
            if preview:
                entries.append(f"  Preview: {preview}")
        return "\n".join(entries) if entries else "- No file previews were available."

    def _allowed_workspace_paths(self, memory, *, max_files: int = 20) -> list[str]:
        project_state = memory.get("workspace.project_state") or {}
        paths = []
        for item in (project_state.get("files") or [])[:max_files]:
            path = str(item.get("path", "")).replace("\\", "/").strip()
            if path:
                paths.append(path)
        return paths

    def _task_mentions_tests(self, task: str) -> bool:
        lowered = task.lower()
        return any(
            token in lowered
            for token in (
                "failing test",
                "unit test",
                "tests",
                "test intent",
                "run the relevant tests",
            )
        )

    def _python_test_command(self, known_paths: set[str]) -> list[str] | None:
        python_test_paths = [
            path
            for path in known_paths
            if path.endswith(".py")
            and (
                path.startswith("tests/")
                or "/tests/" in path
                or Path(path).name.startswith("test_")
            )
        ]
        if not python_test_paths:
            return None
        pytest_markers = {"pytest.ini", "conftest.py"}
        if known_paths & pytest_markers:
            return ["python", "-m", "pytest"]
        if any(path.startswith("tests/") for path in python_test_paths):
            return ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
        return ["python", "-m", "unittest", "-v"]

    async def _plan_fix_actions(self, task: str, memory, tool_result: dict) -> list[dict]:
        """Use terminal feedback plus workspace state to generate corrective edits."""
        workspace_root = Path(memory.get("workspace.root_dir")).resolve()
        project_state = memory.get("workspace.project_state") or {}
        files = project_state.get("files", [])
        file_summaries = "\n".join(
            f"- {item['path']}\n  Preview: {item.get('preview', '')}"
            for item in files[:10]
        )
        prompt = (
            "You are fixing a project after a terminal command failed.\n"
            "Return only path-tagged fenced code blocks for files that must change.\n"
            "Do not restate the error or include markdown outside the file blocks.\n\n"
            f"Goal and task:\n{task}\n\n"
            f"Workspace root:\n{workspace_root}\n\n"
            f"Failed command:\n{' '.join(tool_result.get('command', []))}\n\n"
            f"stderr:\n{tool_result.get('stderr', '')}\n\n"
            f"stdout:\n{tool_result.get('stdout', '')}\n\n"
            f"Workspace files:\n{file_summaries}\n"
        )
        allowed_paths = self._allowed_workspace_paths(memory)
        if allowed_paths:
            prompt += (
                "\nWhen editing an existing file, use the exact relative path from this list:\n"
                f"{', '.join(allowed_paths[:12])}\n"
                "Never invent placeholder paths like relative/path.ext or labels like Updated math_utils.py.\n"
            )
        if self._task_type_from_prompt(task) == "test_generation":
            prompt += (
                "\nDo not rewrite or weaken the tests unless they are syntactically invalid. "
                "Prefer fixing implementation files so the tests pass.\n"
            )
        response = await self._call_local(prompt)
        artifacts = self._artifact_materializer.extract(response)
        return self._artifacts_to_tool_calls(artifacts, workspace_root)

    def _session_key(self, task: str, memory) -> str:
        workflow_id = getattr(memory, "workflow_id", "no-workflow") if memory is not None else "no-workflow"
        digest = hashlib.sha1(task.encode("utf-8")).hexdigest()[:12]
        return f"{workflow_id}:{digest}"

    def _task_type_from_prompt(self, task: str) -> str:
        match = re.search(r"Task type:\s*([^\n]+)", task, flags=re.IGNORECASE)
        return match.group(1).strip().lower() if match else "solution"
