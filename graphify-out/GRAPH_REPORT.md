# Graph Report - .  (2026-06-21)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 96 nodes · 130 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4012ee00`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 13|Community 13]]

## God Nodes (most connected - your core abstractions)
1. `get_validation_transforms()` - 9 edges
2. `train_model()` - 9 edges
3. `FocalLoss` - 9 edges
4. `ImageClassifierDataset` - 7 edges
5. `get_training_transforms()` - 7 edges
6. `predict_image()` - 7 edges
7. `build_resnet50()` - 7 edges
8. `get_dataloaders()` - 6 edges
9. `build_efficientnet()` - 6 edges
10. `Feature Extraction` - 6 edges

## Surprising Connections (you probably didn't know these)
- `test_focal_loss()` --calls--> `FocalLoss`  [INFERRED]
  tests/unit/test_image_classifier.py → src/image_classifier/training/losses.py
- `test_inference_predict_image()` --calls--> `get_validation_transforms()`  [INFERRED]
  tests/unit/test_image_classifier.py → src/image_classifier/data_loading/transforms.py
- `test_inference_predict_image()` --calls--> `build_efficientnet()`  [INFERRED]
  tests/unit/test_image_classifier.py → src/image_classifier/models/efficientnet.py
- `test_model_build_efficientnet()` --calls--> `build_efficientnet()`  [INFERRED]
  tests/unit/test_image_classifier.py → src/image_classifier/models/efficientnet.py
- `test_model_build_resnet()` --calls--> `build_resnet50()`  [INFERRED]
  tests/unit/test_image_classifier.py → src/image_classifier/models/resnet.py

## Import Cycles
- None detected.

## Communities (14 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (15): Compose, get_dataloaders(), ImageClassifierDataset, Initializes datasets and checks paths., Returns (train_loader, val_loader).         Ensures datasets are setup before cr, Convenience function to quickly get dataloaders and class names., Wrapper class to manage PyTorch ImageFolder datasets and DataLoaders.     It exp, get_inverse_transforms() (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.14
Nodes (13): build_resnet50(), Builds a ResNet50 model with a custom classification head.          Args:, Module, Module, test_focal_loss(), test_metrics_get_predictions(), test_model_build_resnet(), generate_classification_report() (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.25
Nodes (9): Any, Module, Tensor, evaluate(), Orchestrates the training process with AMP, Early Stopping and LR Scheduling., train_model(), train_one_epoch(), FocalLoss (+1 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (13): Backpropagation & Optimization, Classification Head, Data Augmentation, EfficientNet, Feature Extraction, Loss Calculation, Raw Image Loader, Resizing & Normalization (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.36
Nodes (6): get_imagenet_classes(), predict_image(), Runs an inference prediction on a single image.          Args:         image_byt, Fallback utility to load standard ImageNet class names if needed., Module, test_inference_predict_image()

### Community 5 - "Community 5"
Cohesion: 0.40
Nodes (4): build_efficientnet(), Builds an EfficientNet-B0 model with a custom classification head.     Efficient, Module, test_model_build_efficientnet()

### Community 6 - "Community 6"
Cohesion: 0.40
Nodes (4): build_vgg16(), Builds a VGG16 model with a custom classification head., Module, test_model_build_vgg16()

## Knowledge Gaps
- **18 isolated node(s):** `run_tests.sh script`, `Module`, `Module`, `Module`, `Module` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_model()` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `get_validation_transforms()` connect `Community 0` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `predict_image()` connect `Community 4` to `Community 1`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `get_validation_transforms()` (e.g. with `test_inference_predict_image()` and `test_transforms_exist()`) actually correct?**
  _`get_validation_transforms()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `FocalLoss` (e.g. with `Any` and `Module`) actually correct?**
  _`FocalLoss` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `run_tests.sh script`, `Wrapper class to manage PyTorch ImageFolder datasets and DataLoaders.     It exp`, `Initializes datasets and checks paths.` to the rest of the system?**
  _35 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.14035087719298245 - nodes in this community are weakly interconnected._