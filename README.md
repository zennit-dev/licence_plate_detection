# ML Model Development Template

A production-ready template for developing machine learning models that can be deployed to web (Laravel/React) and
mobile (React Native) applications. This template provides a standardized structure and workflow for training models and
exporting them for various deployment targets.

## Features

- **Structured Development Workflow**: From data exploration to production deployment
- **Multiple Export Formats**:
    - TensorFlow.js for React/React Native
    - TensorFlow Lite for mobile
    - TensorFlow Serving for REST APIs
- **Development Tools**:
    - Jupyter notebooks for experimentation
    - Type checking and linting
    - Automated testing
    - Experiment tracking
- **Production Ready**:
    - Model versioning
    - Export utilities
    - Performance optimization

## Project Structure

```
.
├── config/                            # Configuration files
├── data/                              # Data files
│   ├── processed/                     # Preprocessed data
│   └── raw/                           # Original data
├── logs/                              # Logs
│   ├── tensorboard/                   # Training logs   
│   └── app.log                        # Application logs
├── models/                            # Models
│   ├── checkpoints/                   # Training checkpoints
│   ├── exported/                      # Production-ready exports
│   │   ├── tfjs/                      # TensorFlow.js models
│   │   ├── tflite/                    # TensorFlow Lite models
│   │   └── serving/                   # TensorFlow Serving models
│   └── saved_models/                  # Saved TensorFlow models
│       └── {version}_{timestamp}/     # Models with version and timestamp
│           ├── metadata.json          # Model metadata
│           ├── model.keras            # Keras model
│           └── model.h5               # Saved model
├── notebooks/                         # Jupyter notebooks
├── src/                               # Source code
│   └── tools/                         # Continuous Integration configs
```

## Setup

1. **Prerequisites**
   ```bash
   # Install Python 3.11+
   python --version  # Should be 3.11 or higher

   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Project Installation**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd ml-course

   # Install dependencies
   poetry install

   # Activate the virtual environment
   poetry shell
   ```

3. **Environment Setup**
   ```bash
   # Run setup script
   ./setup.sh
   ```

## Development Workflow

### 1. Data Exploration and Model Development

Use the provided Jupyter notebooks in `notebooks/`:

- `1_data_exploration.ipynb`: Data analysis and preprocessing
- `2_model_development.ipynb`: Model architecture and training
- `3_model_export.ipynb`: Export models for production

### 2. Training

```bash
# Train a model
poetry run ml train --settings settings/default.py

# Monitor data
tensorboard --logdir logs/tensorboard
```

### 3. Export for Production

#### React/React Native (TensorFlow.js)

```bash
# Export model
poetry run ml export --format tfjs --output models/exported/tfjs

# Usage in React:
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('path/to/model.json');
const prediction = model.predict(tf.tensor(data));
```

#### Mobile (TensorFlow Lite)

```bash
# Export model
poetry run ml export --format tflite --output models/exported/tflite

# The exported model can be used with TensorFlow Lite in mobile apps
```

#### Laravel Backend (TensorFlow Serving)

```bash
# Export model
poetry run ml export --format serving --output models/exported/serving

# Start TensorFlow Serving
docker run -p 8501:8501 \\
    --mount type=bind,source=/path/to/models/exported/serving,target=/models/my_model \\
    -e MODEL_NAME=my_model -t tensorflow/serving

# Make predictions via REST API:
curl -X POST http://localhost:8501/v1/models/my_model/predict \\
    -d '{"instances": [[1.0, 2.0, 3.0]]}'
```

## Development Tools

```bash
# Run all quality checks
poetry run lint

# Run tests
poetry run test

# Clean project
poetry run clean
```

## Best Practices

1. **Version Control**
    - Use semantic versioning for models
    - Tag releases when exporting to production
    - Document model changes

2. **Testing**
    - Write unit tests for preprocessing
    - Test model inputs/outputs
    - Validate exported models

3. **Deployment**
    - Monitor model performance
    - Version API endpoints
    - Implement gradual rollout

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
