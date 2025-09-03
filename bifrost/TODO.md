# Bifrost TODO

## Feature Ideas

### Watch/Sync Mode for Development
Add a `--watch` flag to enable hot-reload development workflow:

**Current pain point:** Each code change requires:
1. `git add -A`
2. `git commit -m "msg"`  
3. `git push`
4. `bifrost deploy ...`

**Proposed solutions:**

1. **File watcher + auto-deploy**
   ```bash
   bifrost deploy --watch SSH_CONNECTION COMMAND
   ```
   - Uses filesystem watchers (watchdog) to detect file changes
   - Auto-commits and pushes git changes
   - Re-runs the command on remote
   
2. **Direct file sync (bypass git)**
   ```bash  
   bifrost deploy --watch --no-git SSH_CONNECTION COMMAND
   ```
   - Uses rsync/scp to sync files directly to remote
   - Much faster than git push/pull cycle
   - Better for rapid iteration
   
3. **Remote filesystem mount**
   - Mount local filesystem on remote via SSHFS
   - Run code directly on mounted files
   - True hot reload with no sync delay

**Implementation notes:**
- Watch common dev file extensions (.py, .js, .ts, etc.)
- Debounce file changes to avoid excessive syncing
- Preserve existing one-shot deploy behavior as default
- Consider excluding large files/directories (.git, node_modules, etc.)

This would reduce the development cycle from 4 commands to 1 initial setup command.