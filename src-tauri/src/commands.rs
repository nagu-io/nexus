use serde_json::Value;

#[tauri::command]
pub async fn check_health() -> Result<Value, String> {
    match reqwest::get("http://127.0.0.1:8000/status").await {
        Ok(res) => {
            let json = res.json::<Value>().await.map_err(|e| e.to_string())?;
            Ok(json)
        },
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub async fn get_ollama_models() -> Result<Vec<String>, String> {
    match reqwest::get("http://127.0.0.1:11434/api/tags").await {
        Ok(res) => {
            let json: Value = res.json().await.map_err(|e| e.to_string())?;
            let mut models = Vec::new();
            if let Some(models_arr) = json.get("models").and_then(|m| m.as_array()) {
                for m in models_arr {
                    if let Some(name) = m.get("name").and_then(|n| n.as_str()) {
                        models.push(name.to_string());
                    }
                }
            }
            Ok(models)
        },
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub async fn pull_model(model: String) -> Result<String, String> {
    // In a real scenario we might stream this, here we just trigger it and return OK when done.
    let client = reqwest::Client::new();
    match client.post("http://127.0.0.1:11434/api/pull")
        .json(&serde_json::json!({ "name": model, "stream": false }))
        .send().await {
        Ok(_) => Ok("Pulled successfully".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
pub async fn run_build(goal: String) -> Result<String, String> {
    let client = reqwest::Client::new();
    match client.post("http://127.0.0.1:8000/chat")
        .json(&serde_json::json!({ "message": goal }))
        .send().await {
        Ok(res) => {
            let body = res.text().await.map_err(|e| e.to_string())?;
            Ok(body)
        },
        Err(e) => Err(e.to_string()),
    }
}
