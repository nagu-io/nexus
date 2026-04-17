pub fn start_python_backend(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let app_dir = app.path_resolver()
        .app_data_dir()
        .expect("failed to get app dir");
    
    // Create the dir if it doesn't exist
    if !app_dir.exists() {
        std::fs::create_dir_all(&app_dir)?;
    }

    // Since NEXUS is expected to be installed via `pip install -e .`
    // We launch it in the daemon mode from working dir.
    // In production we would package it, but for now we follow the script.
    let current_dir = std::env::current_dir().unwrap_or_else(|_| app_dir.clone());

    // Build PYTHONPATH relative to the working directory so the setup is portable
    let nexus_root = current_dir.to_string_lossy().to_string();
    let nexus_pkgs = current_dir.join(".nexus_pkgs").to_string_lossy().to_string();
    let python_path = format!("{};{}", nexus_root, nexus_pkgs);
    
    std::process::Command::new("python")
        .args([
            "-m", "uvicorn",
            "nexus.api:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ])
        .env("PYTHONPATH", &python_path)
        .current_dir(&current_dir)
        .spawn()
        .map_err(|e| format!("failed to start NEXUS backend: {}", e))?;
    
    // Wait for backend to be ready
    std::thread::sleep(std::time::Duration::from_secs(3));
    Ok(())
}
