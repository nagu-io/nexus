"""
Jarvis Interface — interactive chat loop for NEXUS.
Supports text and voice mode.
Maintains session memory via Supabase.
Uses AEON Mind Router for all requests.
"""

import asyncio
import uuid
from rich.console import Console
from rich.panel import Panel

console = Console()

WELCOME_MESSAGE = """
Welcome to NEXUS Jarvis.
Your private AI assistant powered by CompressX + AEON + ReflectScore.

Commands:
  exit / quit    — exit Jarvis
  /clear         — clear session memory
  /status        — show system status
  /reflect       — show last ReflectScore
  /agents        — list available agents

Type anything to chat. Jarvis is listening.
"""


class JarvisInterface:
    """
    Main Jarvis interactive interface.
    Text mode by default, voice mode optional.
    """

    def __init__(self, voice_mode: bool = False):
        self.voice_mode = voice_mode
        self.session_id = str(uuid.uuid4())
        self.last_reflect = None
        self._router = None
        self._memory = None
        self._voice_input = None
        self._voice_output = None
        self._memory_warning_shown = False

    def _get_router(self):
        if not self._router:
            from nexus.router.mind_router import MindRouter
            self._router = MindRouter()
        return self._router

    def _get_memory(self):
        if not self._memory:
            from nexus.memory.supabase_memory import SupabaseMemory
            self._memory = SupabaseMemory()
        return self._memory

    def _get_voice(self):
        if not self._voice_input:
            from nexus.jarvis.voice import VoiceInput, VoiceOutput
            self._voice_input = VoiceInput()
            self._voice_output = VoiceOutput()

    async def run(self):
        """Main Jarvis event loop."""
        console.print(Panel(WELCOME_MESSAGE, title="[bold cyan]NEXUS Jarvis[/bold cyan]", style="cyan"))

        if self.voice_mode:
            self._get_voice()
            console.print("[green]Voice mode active. Speak after the prompt.[/green]")

        memory_status = self._get_memory().status_message()
        if "configured" not in memory_status.lower():
            console.print(f"[yellow]{memory_status}[/yellow]")
            self._memory_warning_shown = True

        while True:
            try:
                # Get input
                if self.voice_mode:
                    user_input = self._voice_input.listen(duration=5)
                    if not user_input:
                        continue
                else:
                    user_input = input("\n[bold cyan]You:[/bold cyan] ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("[cyan]Goodbye.[/cyan]")
                    break

                elif user_input == "/clear":
                    memory = self._get_memory()
                    if "configured" not in memory.status_message().lower():
                        console.print("[yellow]Persistent memory is not enabled, so there is nothing stored to clear.[/yellow]")
                    else:
                        await memory.clear_session(self.session_id)
                        console.print("[green]Session memory cleared.[/green]")
                    continue

                elif user_input == "/status":
                    from nexus.cli import cli
                    console.print("[cyan]Run: nexus status[/cyan]")
                    continue

                elif user_input == "/reflect":
                    if self.last_reflect is not None:
                        score = self.last_reflect["score"]
                        verdict = self.last_reflect["verdict"]
                        action = self.last_reflect["action"]
                        color = "green" if verdict == "clean" else "yellow" if verdict == "warning" else "red"
                        console.print(f"[{color}]Last ReflectScore: {score:.3f} ({verdict} -> {action})[/{color}]")
                        if self.last_reflect.get("warning"):
                            console.print(f"[{color}]{self.last_reflect['warning']}[/{color}]")
                    else:
                        console.print("[dim]No ReflectScore yet.[/dim]")
                    continue

                elif user_input == "/agents":
                    console.print("[cyan]Available agents: coding, research, memory, file, canary[/cyan]")
                    continue

                # Save user message to memory
                await self._get_memory().save_message(self.session_id, "user", user_input)

                # Route through AEON
                console.print("[dim]Thinking...[/dim]")
                router = self._get_router()
                result = await router.route(user_input, return_meta=True)
                response = result["response"]
                self.last_reflect = {
                    "score": result["reflect_score"],
                    "verdict": result["reflect_verdict"],
                    "action": result["reflect_action"],
                    "warning": result["warning"],
                    "was_rerouted": result["was_rerouted"],
                }

                # Save response to memory
                await self._get_memory().save_message(self.session_id, "assistant", response)

                # Output response
                if result["warning"]:
                    warning_style = "yellow" if result["reflect_action"] != "block" else "red"
                    console.print(Panel(result["warning"], title="[bold]ReflectScore[/bold]", style=warning_style))
                reduction = result.get("context_reduction")
                if reduction and reduction.get("reduced"):
                    console.print(
                        f"[dim]Context reduced {reduction['original_length']} -> "
                        f"{reduction['reduced_length']} chars via {reduction['backend']}[/dim]"
                    )
                console.print(Panel(response, title="[bold green]NEXUS[/bold green]", style="green"))

                if self.voice_mode:
                    self._voice_output.speak(response)

            except KeyboardInterrupt:
                console.print("\n[cyan]Interrupted. Type 'exit' to quit.[/cyan]")
            except Exception as error:
                console.print(
                    f"[red]Jarvis hit a recoverable error: {error}. "
                    "You can keep chatting, or run `nexus doctor` if this keeps happening.[/red]"
                )
