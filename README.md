# Few-Shot Visual Object Counting System

A real-time visual object counting system designed for industrial environments, capable of counting arbitrary object categories using a limited number of reference images without requiring model retraining.

## Overview

In manufacturing workflows, accurately counting objects prior to shipment is critical but often inefficient and subject to human error. Manual counting is slow and unreliable, while weight-based estimation is frequently inapplicable for large or irregular objects.

This project proposes a **Few-Shot Visual Object Counting System** that:

- Counts objects in real-time from camera input.
- Supports new object types with a minimal set of reference images (≤10).
- Eliminates the need for retraining when new parts are introduced.
- Enables selective counting per session (e.g., specific monitoring of designated object categories).

## Problem Statement

### Challenges

- **Human Error:** Manual counting is prone to fatigue and distraction.
- **Data Scarcity:** Traditional computer vision requires large, labeled datasets for every new class.
- **Dynamic Environments:** Frequent introduction of new object types in factory settings makes static models obsolete.
- **Physical Constraints:** Large or non-uniform objects cannot be accurately measured via weight-based methods.

### Objective

Design and implement a system capable of accurately counting arbitrary objects in real-time using only a few reference images, maintaining high precision without model retraining.

## Technical Approach

The system integrates object detection, feature embedding (few-shot learning), and multi-object tracking to follow a **Detect → Identify → Track → Count** pipeline.

## System Architecture

1.  **Reference Processing:**
    - Reference Images → Feature Encoder (DINOv2) → Prototype Embedding.
2.  **Inference Pipeline:**
    - Video Input → Object Detection (YOLOv8) → Object Cropping.
    - Cropped Objects → Feature Encoding (DINOv2) → Similarity Matching (Cosine Similarity).
    - Matched Objects → Object Tracking (ByteTrack) → Counting Logic (Unique ID Management).
    - Final Output → Real-time Count.

## Operational Workflow

### Phase 0: Reference Setup

Users provide 3–10 images per object category. The system encodes these into a prototype embedding representing the "ideal" features of that class.

### Phase 1: Detection

Candidate objects are detected using YOLOv8, which extracts bounding boxes for all potential items in the frame.

### Phase 2: Feature Matching

Detected objects are cropped and encoded using DINOv2. These embeddings are compared against the reference prototypes using cosine similarity to determine the object class.

### Phase 3: Tracking

The system assigns unique IDs using ByteTrack, allowing it to maintain the identity of objects across consecutive frames and handle temporary occlusions.

### Phase 4: Counting

The counting logic increments the total only when a new, unique track ID is confirmed, preventing duplicate counts of the same physical object.

## Core Components

| Component         | Technology        | Primary Function                    |
| ----------------- | ----------------- | ----------------------------------- |
| Object Detection  | YOLOv8            | High-speed object localization      |
| Feature Encoder   | DINOv2            | Few-shot representation             |
| Tracking          | ByteTrack         | Temporal object consistency         |
| Similarity Metric | Cosine Similarity | Feature matching and classification |

## Implementation Details

### Counting Logic

```python
if track_id not in counted_ids:
    count += 1
    counted_ids.add(track_id)
```

### Performance Optimization

- **Execution Frequency:** Detection runs every N frames (e.g., every 5 frames) to reduce computational load.
- **Interpolation:** Tracking is utilized to maintain state in intermediate frames.
- **Efficiency:** Implementation of batch feature encoding and optimized image resizing for low-latency inference.

## Evaluation Metrics

### Counting Accuracy

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**

### Detection Quality

- **Precision / Recall**
- **mAP (mean Average Precision)**

### System Efficiency

- **Latency (ms/frame)**
- **Throughput (FPS)**

## Proposed Experiments

- Comparative analysis of 1-shot vs. 5-shot vs. 10-shot performance.
- Ablation studies: Performance with and without tracking modules.
- Benchmarking against human counting and fully supervised detection models.

## Key Features

- **Few-Shot Capability:** Adaptable to new categories without retraining.
- **Real-Time Processing:** Optimized for industrial throughput.
- **Configurability:** Supports multi-object selection per session.
- **Scalability:** Designed for deployment in diverse industrial use cases.

## Constraints and Limitations

- Highly similar objects (intra-class variance) may lead to classification errors.
- Severe occlusion can degrade detection and tracking performance.
- Tracking ID switches in crowded scenes may impact count consistency.

## Future Directions

- Integration of active learning with human-in-the-loop feedback.
- Deployment to edge devices (e.g., NVIDIA Jetson).
- Advanced occlusion handling algorithms.
- Temporal consistency modeling for improved robustness.

## Project Structure

```text
project/
├── data/           # Dataset storage
├── models/         # Pre-trained weights
├── reference/      # Reference image prototypes
├── src/            # Source code
│   ├── detector.py
│   ├── encoder.py
│   ├── tracker.py
│   ├── matcher.py
│   ├── counter.py
│   └── main.py
├── utils/          # Helper functions
├── configs/        # System configurations
└── README.md       # Project documentation
```

## Conclusion

This system provides a scalable approach to object counting in dynamic industrial environments. By leveraging few-shot learning and robust tracking, it addresses the critical needs of adaptability, low latency, and minimal data requirements.

## License

MIT License
