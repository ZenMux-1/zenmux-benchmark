# GitHub Actions Setup

This document describes the configuration needed for the GitHub Actions workflows in this repository.

## Required GitHub Secrets

To use the GitHub Actions workflows, you need to configure the following secrets in your repository settings:

### Required Secrets

1. **ZENMUX_API_KEY** (Required)
   - Your ZenMux API key for accessing the unified API
   - Get this from your ZenMux account dashboard

2. **HF_TOKEN** (Required for gated datasets)
   - Your Hugging Face access token for accessing gated datasets like 'cais/hle'
   - Get this from [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - **Important**: You must also request access to the dataset through the Hugging Face website first

### Optional Environment Secrets

3. **ZENMUX_BASE_URL** (Optional)
   - Custom ZenMux base URL if different from default
   - Default: `https://zenmux.ai/api/v1`

4. **ZENMUX_API_BASE_URL** (Optional)
   - Alternative API base URL configuration
   - Only needed if using a custom endpoint

## How to Set GitHub Secrets

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with the name and value from the list above

## Workflows Available

### 1. Main Benchmark Workflow (`benchmark.yml`)

**Purpose**: Run comprehensive AI model benchmarks

**Features**:
- All evaluation modes (all/single/filter)
- Model exclusion support
- Text-only mode and sample limits
- Automatic commit and push of results (optional)

**Usage**:
- Go to **Actions** tab in your repository
- Select "ZenMux HLE Benchmark"
- Click "Run workflow"
- Configure parameters as needed

### 2. Fix Benchmark Workflow (`fix-benchmark.yml`)

**Purpose**: Fix failed evaluations from previous runs

**Features**:
- Downloads previous results from artifacts
- Fixes failed evaluations and judgments
- Automatic commit and push of fixed results (optional)

**Usage**:
- Go to **Actions** tab in your repository
- Select "ZenMux HLE Benchmark Fix"
- Click "Run workflow"
- Enter the timestamp directory to fix

## Git Commits

Both workflows can automatically commit and push results to the repository:

- **Benchmark workflow**: Set `commit_results` in advanced_options or keep default
- **Fix workflow**: Enable "Commit and push fixed results" option (default: enabled)

The workflows will:
1. Create descriptive commit messages with run details
2. Add results/ and logs/ directories
3. Push to the current branch
4. Use `github-actions[bot]` as the committer

## Troubleshooting

### Permission Issues

If you get permission errors during git push:

1. Ensure repository settings allow GitHub Actions to write
2. Check that the workflow has `contents: write` permission (already configured)
3. Verify `GITHUB_TOKEN` has appropriate permissions

### API Key Issues

If you get authentication errors:

1. Verify `ZENMUX_API_KEY` is correctly set in repository secrets
2. Check that the API key is valid and not expired
3. Ensure the API key has appropriate permissions for model access

### Hugging Face Dataset Access Issues

If you get "Dataset 'cais/hle' is a gated dataset" errors:

1. **Create HF Token**: Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. **Create New Token**: Click "New token" and select "Read" permissions (or "Write" if needed)
3. **Copy Token**: Copy the generated token value
4. **Add to GitHub Secrets**: Add as `HF_TOKEN` in repository secrets
5. **Request Dataset Access**: Visit the [cais/hle dataset page](https://huggingface.co/datasets/cais/hle) and request access
6. **Wait for Approval**: Wait for the dataset owners to approve your access request

**Important Notes**:

- You need BOTH the token AND approval from dataset owners
- Access requests can only be made through the Hugging Face website while logged in
- Some gated datasets may require "Write" permissions even for read operations

### Environment Issues

If you get environment-related errors:

1. Check that optional environment variables are correctly set if needed
2. Verify custom URLs are accessible from GitHub Actions runners
3. Review the workflow logs for specific error messages
