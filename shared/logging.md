# Modern Python Logging - Key Lessons

Based on "Modern Python logging" video by James Murphy (mCoding).

## Main Principles

### 1. Stop using print(), start using logging
- Replace `print()` statements with proper logging calls
- Use `logging.getLogger(__name__)` instead of direct logging functions

### 2. Avoid basic logging setup for complex cases
- Don't use `logging.basicConfig()` beyond simple scenarios
- Use `logging.config.dictConfig()` for proper configuration

### 3. Understanding logging architecture
- **Loggers**: What you use in code (`logger.info()`, `logger.debug()`)
- **Handlers**: Where logs go (stdout, files, email, services)  
- **Formatters**: How log records become text
- **Filters**: Optional message modification/filtering

### 4. Recommended configuration approach
- Put all handlers on the root logger (not individual loggers)
- Use propagation to send all messages up to root handlers
- Avoid complex handler hierarchies unless needed
- Ensures third-party library logs are formatted consistently

### 5. Best practices
- Always use named loggers: `logger = logging.getLogger(__name__)`
- Don't use top-level logging functions like `logging.info()`
- Configure once at application startup
- Let messages propagate up to root handlers

## Implementation for this monorepo

Our standardized approach follows these principles:
- Centralized dict config in `shared/logging_config.py`
- Environment variable control (`LOG_LEVEL`)
- All handlers on root logger for consistency
- Named loggers throughout codebase