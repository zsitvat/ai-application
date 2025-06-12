# GitHub Actions CI/CD

This repository uses GitHub Actions for continuous integration and deployment.
## Workflows

### 1. Docker CI/CD (`docker-ci.yml`)
- **Triggers**: Push to `development`, `test`, `main` branches, manual trigger
- **Features**:
  - Builds and pushes Docker images to GitHub Container Registry (ghcr.io)
  - Supports multi-platform builds (linux/amd64, linux/arm64)
  - Uses Docker layer caching for faster builds
  - Tags images appropriately based on branch/trigger type
- **Manual Execution**: Can be triggered manually for any branch

### 2. Test and Lint (`test-lint.yml`)
- **Triggers**: Push and PR to `development`, `test`, `main` branches
- **Features**:
  - Runs tests with pytest
  - Code coverage reporting
  - Linting with flake8, black, isort
  - Type checking with mypy
  - Uses Poetry for dependency management
  - Redis service for integration tests

### 3. Security and Dependencies (`security.yml`)
- **Triggers**: Push, PR, and weekly schedule
- **Features**:
  - Security vulnerability scanning with safety
  - Security linting with bandit
  - Dependency review for PRs
  - Automated dependency updates (weekly)

### 4. Release (`release.yml`)
- **Triggers**: Git tags matching `v*.*.*` pattern, manual trigger
- **Features**:
  - Automated release creation
  - Changelog generation
  - Multi-platform Docker image builds
  - Semantic versioning support
  - Optional staging deployment

## Repository Settings

### Required Secrets
No additional secrets are required - the workflows use the built-in `GITHUB_TOKEN` for authentication.

### Container Registry
Images are pushed to GitHub Container Registry (`ghcr.io`) and are publicly accessible.

### Branch Protection
Recommended branch protection rules for `main` branch:
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include administrators in restrictions

## Usage

### Running Tests Locally
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run black src
poetry run flake8 src
poetry run isort src
poetry run mypy src
```

### Building Docker Images Locally
```bash
# Build image
docker build -t recruiter-ai-app .

# Run container
docker run -p 5000:5000 recruiter-ai-app
```

### Creating Releases
1. **Automatic**: Push a git tag with semantic versioning:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Manual**: Use the GitHub Actions UI to trigger the release workflow with a custom version.

### Accessing Container Images
```bash
# Pull latest image
docker pull ghcr.io/username/repository:latest

# Pull specific version
docker pull ghcr.io/username/repository:v1.0.0
```

## Migration from GitLab CI

The GitHub Actions workflows provide equivalent functionality to the GitLab CI configuration:

| GitLab CI Job | GitHub Actions Equivalent | Changes |
|---------------|---------------------------|---------|
| `docker-build` | `docker-ci.yml` job `docker-build` | Uses GitHub Container Registry instead of custom registry |
| `docker-tag` | `docker-ci.yml` job `docker-tag-branch` | Automatic tagging based on branch |
| `docker-tag-main` | `docker-ci.yml` job `docker-tag-main` | Tags main branch as `master` |
| Manual triggers | `workflow_dispatch` | Available on all workflows |

### Key Improvements
- **Multi-platform builds**: Support for both AMD64 and ARM64 architectures
- **Enhanced security**: Automated security scanning and dependency updates
- **Better caching**: Docker layer caching and Poetry dependency caching
- **Comprehensive testing**: Automated testing with coverage reporting
- **Release automation**: Automated changelog generation and release creation

## Troubleshooting

### Common Issues
1. **Build failures**: Check the Actions tab for detailed logs
2. **Permission issues**: Ensure proper repository permissions for packages
3. **Test failures**: Run tests locally first to debug issues

### Getting Help
- Review workflow logs in the GitHub Actions tab
- Check the Poetry and Docker documentation for dependency issues
- Open an issue for workflow-specific problems
