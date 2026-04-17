#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod setup;
mod health;
mod commands;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            setup::start_python_backend(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::check_health,
            commands::get_ollama_models,
            commands::pull_model,
            commands::run_build,
        ])
        .run(tauri::generate_context!())
        .expect("error while running NEXUS");
}
