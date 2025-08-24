# R.E.D. Framework: Recursive Expert Delegation

A novel semi-supervised text classification framework that combines traditional machine learning with Large Language Model (LLM) validation to achieve human-expert level performance on large-scale classification tasks.

## Overview

R.E.D. (Recursive Expert Delegation) is designed to solve the challenging problem of text classification when you have:

- **Large number of classes** (100-1000+)
- **Limited training data** per class (30-100 samples)
- **High accuracy requirements** (90%+ precision)
- **Cost and speed constraints**

```mermaid
graph TB
    A["📊 Large-Scale Text Classification Challenge"] --> B["🎯 R.E.D. Framework"]
  
    B --> C["🔄 Three-Stage Process"]
  
    C --> D["1️⃣ Greedy Subset Selection<br/>📦 Split N classes → n subsets"]
    C --> E["2️⃣ Semi-Supervised Classification<br/>🤖 Train lightweight classifiers"]
    C --> F["3️⃣ LLM Proxy Validation<br/>🧠 Expert-level validation"]
  
    D --> G["📈 Reduced Complexity"]
    E --> H["⚡ Fast Processing"]
    F --> I["🎯 High Accuracy"]
  
    G --> J["✅ Scalable Solution"]
    H --> J
    I --> J
  
    style A fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style B fill:#e8f5e8,stroke:#388e3c,stroke-width:3px
    style J fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style D fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style E fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style F fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

## How It Works

R.E.D. employs a three-stage approach:

### 1. **Greedy Subset Selection**

- Partitions large label spaces into smaller, manageable subsets
- Uses embedding-based similarity to ensure each subset contains maximally dissimilar labels
- Reduces complexity from N-class to multiple n-class problems (where n << N)

```mermaid
graph LR
    A["📋 Large Label Set<br/>Classes: 1000+"] --> B["🔍 Embedding Analysis"]
  
    B --> C["📊 Similarity Matrix<br/>Calculate distances<br/>between all labels"]
  
    C --> D["🎯 Greedy Selection<br/>Pick maximally<br/>dissimilar labels"]
  
    D --> E["📦 Subset 1<br/>Classes: 1-8"]
    D --> F["📦 Subset 2<br/>Classes: 9-16"]
    D --> G["📦 Subset 3<br/>Classes: 17-24"]
    D --> H["📦 ... More Subsets"]
  
    I["⚡ Benefits"] --> J["Reduced Complexity<br/>N → n problems"]
    I --> K["Better Accuracy<br/>Less confusion<br/>between classes"]
    I --> L["Parallel Processing<br/>Independent<br/>classifiers"]
  
    style A fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style D fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style E fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style F fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style G fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style H fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style I fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

### 2. **Semi-Supervised Classification with Noise Oversampling**

- Trains lightweight classifiers for each subset
- Includes "noise" class with oversampled data from other subsets
- Focuses on identifying samples that need validation rather than perfect classification

```mermaid
graph TB
    A["📦 Label Subset<br/>Classes A, B, C"] --> B["🎯 Training Data"]
  
    B --> C["✅ Positive Examples<br/>Class A: 50 samples<br/>Class B: 45 samples<br/>Class C: 55 samples"]
  
    D["📦 Other Subsets<br/>Classes D,E,F + G,H,I..."] --> E["🔀 Noise Sampling<br/>Random samples from<br/>other categories"]
  
    E --> F["⚫ Noise Class<br/>300 samples<br/>(2x oversampling)"]
  
    C --> G["🤖 Lightweight Classifier<br/>Logistic Regression<br/>or Random Forest"]
    F --> G
  
    G --> H["🎯 Predictions"]
  
    H --> I["✅ In-Subset<br/>Classes A, B, C"]
    H --> J["⚫ Noise<br/>Not in this subset"]
  
    K["💡 Key Insight"] --> L["Focus on Detection<br/>Not Perfect Classification"]
    K --> M["Fast Training<br/>Lightweight models"]
    K --> N["Noise Awareness<br/>Knows what doesn't<br/>belong here"]
  
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style F fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style G fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style I fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style J fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style K fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

### 3. **Proxy Active Learning via LLM**

- Uses LLMs as domain experts to validate classifier predictions
- Dynamically sources similar examples for context-rich validation prompts
- Recursively retrains classifiers with validated samples
- Continues until convergence or saturation

```mermaid
graph TB
    A["📝 Unlabeled Text"] --> B["🤖 Subset Classifiers<br/>Make predictions"]
  
    B --> C["🎯 Uncertainty Sampling<br/>Select most uncertain<br/>predictions"]
  
    C --> D["🔍 Similar Examples<br/>Find similar texts<br/>from training data"]
  
    D --> E["📝 Rich Context Prompt<br/>Label: 'Product Review'<br/>Examples: [similar texts]<br/>Question: Does this text<br/>belong to this label?"]
  
    E --> F["🧠 LLM Expert<br/>GPT-4, Claude, etc.<br/>Acts as domain expert"]
  
    F --> G{"✅ Valid?"}
  
    G -->|Yes| H["✅ Add to Training<br/>High-confidence label"]
    G -->|No| I["❌ Reject<br/>Poor prediction"]
  
    H --> J["🔄 Retrain Classifiers<br/>With new validated data"]
    I --> J
  
    J --> K{"🎯 Converged?"}
  
    K -->|No| A
    K -->|Yes| L["🏁 Final Model<br/>Ready for production"]
  
    M["💰 Cost Efficiency"] --> N["Only validate<br/>uncertain samples"]
    M --> O["Reuse similar<br/>examples"]
    M --> P["Iterative improvement<br/>vs. labeling everything"]
  
    style A fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style F fill:#e8f5e8,stroke:#388e3c,stroke-width:3px
    style H fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style I fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style L fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style M fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setup

1. **Clone the repository:**

```bash
git clone <repository-url>
cd recursive_expert_delegation
```

2. **Install dependencies:**

```bash
uv sync --all-groups
```

3. **Activate the environment:**

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Environment Setup

### LLM API Configuration

R.E.D. framework supports multiple LLM providers. You'll need API keys for the models you want to use.

1. **Copy the example environment file:**

```bash
cp .example.env .env
```

2. **Choose your LLM provider(s) and get API keys:**

#### 🤖 **Google AI Studio (Gemini Models)** - Recommended for beginners

- **Get API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Models**: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro-exp`
- **Cost**: Free tier available
- **Add to .env**: `GOOGLE_AI_STUDIO_API_KEY=AIzaSy-your-actual-key-here`

#### 🌐 **OpenRouter (Multiple Models)** - Best value

- **Get API Key**: Visit [OpenRouter](https://openrouter.ai/keys)
- **Models**: `glm-4.5-air` (free), `deepseek-r1-0528` (free), `qwen3-30b-a3b` (free)
- **Cost**: Many free models available
- **Add to .env**: `OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here`

#### 🏠 **Ollama (Local Models)** - Privacy focused

- **Installation**: Visit [Ollama.ai](https://ollama.ai/) and install locally
- **Models**: `qwen3-8b`, `phi4-mini`, `deepseek-r1-8b`
- **Cost**: Free (runs on your hardware)
- **Setup**:
  ```bash
  # Install Ollama
  curl -fsSL https://ollama.ai/install.sh | sh

  # Pull a model (example)
  ollama pull deepseek-r1:8b
  ```

#### 🧠 **Anthropic Claude** - Highest quality

- **Get API Key**: Visit [Anthropic Console](https://console.anthropic.com/)
- **Models**: `claude-3.5-sonnet`, `claude-3.7-sonnet`
- **Cost**: Pay-per-use
- **Add to .env**: `ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here`

#### 🔍 **Perplexity AI** - Web-enhanced

- **Get API Key**: Visit [Perplexity Settings](https://www.perplexity.ai/settings/api)
- **Models**: `sonar`, `sonar-small`
- **Cost**: Pay-per-use
- **Add to .env**: `PERPLEXITY_API_KEY=pplx-your-actual-key-here`

3. **Test your configuration:**

```bash
# Quick test with your configured model
python -c "
from config import get_config
config = get_config()
print(f'Default model: {config.default_model}')
"
```

### Model Recommendations by Use Case

```mermaid
graph TD
    A["🎯 Choose Your LLM"] --> B{"💰 Budget?"}
  
    B -->|Free| C["🆓 Free Options"]
    B -->|Paid| D["💳 Paid Options"]
  
    C --> E["🌐 OpenRouter<br/>glm-4.5-air (free)<br/>deepseek-r1-0528 (free)"]
    C --> F["🤖 Google AI Studio<br/>gemini-2.0-flash<br/>(free tier)"]
    C --> G["🏠 Ollama<br/>deepseek-r1:8b<br/>(local, private: works best with <=3 examples/class)"]
  
    D --> H["🧠 Claude 3.7 Sonnet<br/>(highest quality)"]
    D --> I["🤖 Gemini 2.5 Pro<br/>(good balance)"]
    D --> J["🔍 Perplexity Sonar<br/>(web-enhanced : best option for 'updated' learning)"]
  
    style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style F fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style G fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style H fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style I fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style J fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
```

## Quick Start

```mermaid
graph LR
    A["📊 Your Data<br/>CSV/JSON/Pickle<br/>text,label"] --> B["🚀 Initial Training<br/>run_initial_training.py"]
  
    B --> C["📦 Subset Creation<br/>Split labels into<br/>manageable groups"]
  
    C --> D["🤖 Classifier Training<br/>Train models for<br/>each subset"]
  
    D --> E["💾 Save Components<br/>Models + metadata<br/>to output directory"]
  
    E --> F["📝 Unlabeled Data<br/>New texts to<br/>classify"]
  
    F --> G["🔁 Active Learning<br/>run_active_learning.py"]
  
    G --> H["🎯 Uncertainty Sampling<br/>Find uncertain<br/>predictions"]
  
    H --> I["🧠 LLM Validation<br/>Expert validation<br/>of samples"]
  
    I --> J["🔄 Iterative Improvement<br/>Retrain with<br/>validated data"]
  
    J --> K{"🎯 Converged?"}
  
    K -->|No| H
    K -->|Yes| L["✅ Production Ready<br/>High-accuracy<br/>classifier"]
  
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style G fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style I fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style L fill:#e8f5e8,stroke:#388e3c,stroke-width:3px
```

### 1. Prepare Your Data

Your training data should be in CSV, JSON, or pickle format with text and label columns:

```csv
text,label
"This product has amazing battery life!",product_review
"Scientists discover new particle in CERN experiment",news_science
"How to configure SSL certificates in Apache",technical_documentation
...
```

### 2. Initial Training

Run the initial training pipeline to create subsets and train classifiers:

```bash
python src/scripts/run_initial_training.py \
  --seed-data data/train.csv \
  --output-dir ./outputs
```

### 3. Active Learning

Process unlabeled data using the trained system:

```bash
python src/scripts/run_active_learning.py \
  --components-dir ./outputs/components \
  --unlabeled-data data/unlabeled.csv \
  --max-iterations 10
```

## Configuration

The framework uses YAML configuration files for all settings:

- **`src/red/config/main_config.yaml`**: Main framework settings
- **`src/red/config/prompts.yaml`**: LLM prompt templates

### Key Configuration Options

```yaml
# Subset creation
subsetting:
  subset_size: 8
  use_umap: true
  
# Classifier settings
classifier:
  type: "random_forest"
  use_embeddings: true
  noise_oversample_factor: 2.0

# LLM validation
llm_validation:
  model_name: "glm-4.5-air"
  temperature: 0.0
  confidence_threshold: 0.5

# Active learning
active_learning:
  batch_size: 100
  samples_per_iteration: 50
  max_iterations: 10
```

## Architecture

```mermaid
graph TB
    subgraph "🏗️ R.E.D. Framework Architecture"
        subgraph "⚙️ Core Algorithms"
            A["📊 subsetter.py<br/>Label subset creation"]
            B["🤖 classifier.py<br/>Noise-oversampled<br/>classification"]
            C["🧠 validator.py<br/>LLM-based validation"]
        end
    
        subgraph "🔄 Pipelines"
            D["🚀 initial_training.py<br/>Setup pipeline"]
            E["🔁 active_learning.py<br/>Main learning loop"]
        end
    
        subgraph "💾 Data Management"
            F["📁 data_manager.py<br/>I/O and semantic search"]
        end
    
        subgraph "🛠️ Utilities"
            G["💬 llm.py<br/>LLM client"]
            H["⚙️ model_config.py<br/>Model configuration"]
            I["🔍 embeddings.py<br/>Embedding provider"]
        end
    
        subgraph "📋 Configuration"
            J["📝 main_config.yaml<br/>Main settings"]
            K["💭 prompts.yaml<br/>LLM prompts"]
            L["🔧 config_loader.py<br/>Config management"]
        end
    end
  
    %% Data flow connections
    D --> A
    D --> B
    D --> F
    E --> B
    E --> C
    E --> F
  
    B --> I
    C --> G
    C --> I
    F --> I
  
    G --> H
    C --> K
    A --> J
    B --> J
  
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style D fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style F fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style G fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style H fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style I fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style J fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style K fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style L fill:#ffebee,stroke:#d32f2f,stroke-width:2px
```

### File Structure

```
src/red/
├── core/                    # Core algorithms
│   ├── subsetter.py        # Label subset creation
│   ├── classifier.py       # Noise-oversampled classification
│   └── validator.py        # LLM-based validation
├── pipelines/              # Orchestration
│   ├── initial_training.py # Setup pipeline
│   └── active_learning.py  # Main learning loop
├── data/                   # Data management
│   └── data_manager.py     # I/O and semantic search
├── utils/                  # Utilities
│   ├── llm.py             # LLM client
│   ├── model_config.py    # Model configuration
│   └── embeddings.py      # Embedding provider
└── config/                 # Configuration
    ├── main_config.yaml    # Main settings
    ├── prompts.yaml        # LLM prompts
    └── config_loader.py    # Config management
```

## Advanced Usage

### Custom Configuration

Create your own configuration files:

```bash
python src/scripts/run_initial_training.py \
  --seed-data data/train.csv \
  --config-dir ./my_config \
  --subset-size 10 \
  --llm-model "glm-4.5-air"
```

### Verbose Output

Enable detailed logging:

```bash
python src/scripts/run_initial_training.py \
  --seed-data data/train.csv \
  --verbose
```

### Resume Training

Resume active learning from a checkpoint:

```bash
python src/scripts/run_active_learning.py \
  --components-dir ./outputs/components \
  --unlabeled-data data/unlabeled.csv \
  --resume-from ./outputs/checkpoint_iter_5
```

## Supported Models

> 💡 **Tip**: See the [Environment Setup](#environment-setup) section for detailed API key setup instructions.

### LLM Models

| Provider                      | Models                                                                              | Cost                | Setup Required                                     |
| ----------------------------- | ----------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------- |
| **🌐 OpenRouter**       | `glm-4.5-air` (free)`deepseek-r1-0528` (free)`qwen3-30b-a3b` (free) | Free & Paid tiers   | [Get API Key](https://openrouter.ai/keys)             |
| **🤖 Google AI Studio** | `gemini-2.0-flash``gemini-2.5-flash``gemini-2.5-pro-exp`            | Free tier available | [Get API Key](https://aistudio.google.com/app/apikey) |
| **🏠 Ollama**           | `deepseek-r1:8b``qwen3:8b``phi4-mini:latest`                        | Free (local)        | [Install Ollama](https://ollama.ai/)                  |
| **🧠 Anthropic**        | `claude-3.5-sonnet``claude-3.7-sonnet`                                     | Pay-per-use         | [Get API Key](https://console.anthropic.com/)         |
| **🔍 Perplexity**       | `sonar``sonar-small`                                                       | Pay-per-use         | [Get API Key](https://www.perplexity.ai/settings/api) |

### Embedding Models

- **Sentence Transformers**: All [supported models](https://www.sbert.net/docs/pretrained_models.html)
- **Default**: `all-MiniLM-L6-v2` (good balance of speed/quality)
- **Recommended for accuracy**: `all-mpnet-base-v2`
- **Recommended for speed**: `all-MiniLM-L6-v2`

### Model Selection Guide

```mermaid
graph LR
    A["🎯 Your Priority"] --> B["💰 Cost"]
    A --> C["🔒 Privacy"]
    A --> D["⚡ Speed"]
    A --> E["🎯 Accuracy"]
  
    B --> F["OpenRouter<br/>Free models"]
    C --> G["Ollama<br/>Local models"]
    D --> H["Gemini Flash<br/>Fast responses"]
    E --> I["Claude Sonnet<br/>Highest quality"]
  
    style F fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style G fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style H fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style I fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
```

## Performance

R.E.D. has been tested on datasets with:

- **Up to 1,000 classes**
- **10,000-100,000 samples**
- **Achieving 90%+ agreement with human experts**
- **Significant cost reduction vs. pure LLM approaches**

## Examples

### E-commerce Product Classification

```bash
# Initial training with 500 product categories
python src/scripts/run_initial_training.py \
  --seed-data ecommerce_train.csv \
  --text-column "product_description" \
  --label-column "category" \
  --subset-size 12

# Process 50K unlabeled products
python src/scripts/run_active_learning.py \
  --components-dir ./outputs/components \
  --unlabeled-data ecommerce_unlabeled.csv \
  --batch-size 200 \
  --samples-per-iteration 100
```

### Scientific Paper Classification

```bash
# Train on academic abstracts
python src/scripts/run_initial_training.py \
  --seed-data papers_train.csv \
  --text-column "abstract" \
  --label-column "field" \
  --llm-model "claude-3.5-sonnet" \
  --embedding-model "all-mpnet-base-v2"
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Style

```bash
uv run black src/
uv run flake8 src/
```

### Adding New LLM Providers

1. Extend `src/red/utils/model_config.py`
2. Add configuration in `utils/model_configs.py`
3. Update prompt formatting if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use R.E.D. in your research, please cite:

```bibtex
@article{red2025,
  title={R.E.D.: Scaling Text Classification with Expert Delegation},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]

## Roadmap

- [ ] Support for multilingual classification
- [ ] Integration with more LLM providers
- [ ] Automated hyperparameter tuning
- [ ] Web interface for easier usage
- [ ] Performance optimization for very large datasets
