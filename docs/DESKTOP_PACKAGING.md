# NEXUS Desktop Packaging

NEXUS now has a dedicated Electron packaging surface in [`desktop/package.json`](/D:/nexus/desktop/package.json).

## Build Flow

1. Build the React desktop UI:
   - `npm --prefix dashboard run build`
2. Bundle the FastAPI backend sidecar:
   - `python tools/build_backend_bundle.py`
3. Package the installer:
   - `npm --prefix desktop run dist`

## What Gets Bundled

- Electron shell: [`desktop/main.cjs`](/D:/nexus/desktop/main.cjs) and [`desktop/preload.cjs`](/D:/nexus/desktop/preload.cjs)
- Dashboard bundle: `dashboard/dist`
- Backend sidecar: `desktop/.artifacts/backend/nexus-backend`
- Default local adapter pack: `lora_model`

## Packaged Runtime Behavior

- User-writable settings are stored via `NEXUS_ENV_PATH` in the Electron `userData` directory, not inside the app resources.
- Packaged builds auto-detect the bundled adapter pack and default to adapter mode when no override exists.
- The backend receives `NEXUS_APP_ROOT` so model-control surfaces can inspect packaged resources instead of assuming a dev repo checkout.

## Notes

- `CompressX` mock outputs are surfaced in the desktop UI but do **not** count as real sub-1GB packaging proof.
- The backend bundle script requires `PyInstaller` in the active Python environment.
