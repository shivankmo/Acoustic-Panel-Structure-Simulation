# Acoustic Panel Structure Simulation

A computational framework for designing and optimizing acoustic panels using finite element simulation, machine learning surrogate models, and multi-objective optimization. This project generates diverse panel geometries, simulates their acoustic performance, and identifies optimal designs that balance acoustic efficiency, cost, and sustainability.

## Overview

This project implements a complete workflow for acoustic panel design optimization:

1. **Dataset Generation**: Create diverse 3D panel geometries with varying shapes, materials, and properties
2. **Finite Element Simulation**: Simulate acoustic performance using FEniCS
3. **Surrogate Model Development**: Train machine learning models to predict acoustic metrics
4. **Multi-Objective Optimization**: Use NSGA-II to find optimal designs
5. **Results and Analysis**: Visualize and analyze performance trends

## Features

- **Geometric Diversity**: Generates panels with varying complexity, thickness, hole patterns, curvature, and porosity
- **Material Options**: Supports foam, cork, recycled plastic, and cardboard with full material property modeling
- **Acoustic Metrics**: Computes Transmission Loss (TL), Absorption Coefficient, and Noise Reduction Coefficient (NRC)
- **Multi-Objective Optimization**: Balances acoustic performance, cost, and sustainability
- **Surrogate Modeling**: Fast predictions without expensive FEM simulations

## Requirements

### Core Dependencies

- Python 3.7+
- NumPy
- Pandas
- Shapely
- PyMesh
- FEniCS
- scikit-learn
- pymoo
- Matplotlib

### Installation

```bash
# Install Python dependencies
pip install numpy pandas shapely scikit-learn pymoo matplotlib

# Install PyMesh (see PyMesh documentation for installation)
# PyMesh requires additional dependencies and may need to be built from source

# Install FEniCS (see FEniCS documentation for installation)
# FEniCS installation varies by platform
```

## Project Structure

```
Acoustic-Panel-Structure-Simulation/
├── README.md
├── LICENSE
├── src/
│   ├── dataset_generation.py    # Step 1: Geometry and dataset generation
│   ├── fem_simulation.py        # Step 2: FEniCS acoustic simulation
│   ├── surrogate_model.py       # Step 3: ML model training
│   ├── optimization.py          # Step 4: NSGA-II optimization
│   └── visualization.py         # Step 5: Results analysis and plotting
├── data/
│   ├── geometries/              # Generated 3D geometries
│   ├── meshes/                  # Tetrahedral meshes
│   ├── simulation_results.csv   # FEM simulation outputs
│   └── optimized_designs.csv    # Final optimized designs
├── models/
│   └── surrogate_model.pkl      # Trained Random Forest model
└── results/
    └── plots/                   # Visualization outputs
```

## Usage

### Step 1: Dataset Generation

Generate diverse 3D panel geometries with varying properties.

**Key Steps:**
1. Use Shapely to generate 2D cross-sections
2. Extrude 2D shapes into 3D geometries using PyMesh
3. Vary parameters:
   - Shape complexity
   - Thickness
   - Hole radius and spacing
   - Curvature
   - Porosity
4. Maintain consistent scale: 0.5 × 0.5 × 0.02 m
5. Assign materials: foam, cork, recycled plastic, or cardboard
6. Include material properties: density, Young's modulus, Poisson's ratio, cost, sustainability index
7. Compute geometric features: surface area, perimeter, curvature, porosity
8. Apply constraints:
   - 50 ≤ A/t ≤ 300 (area-to-thickness ratio)
   - Total cost ≤ maximum cost
   - Sustainability index ≥ baseline threshold

**Example:**
```python
python src/dataset_generation.py --num_samples 1000 --output_dir data/geometries
```

### Step 2: Finite Element Simulation

Simulate acoustic performance using FEniCS.

**Key Steps:**
1. Convert 3D geometries to tetrahedral meshes using PyMesh
2. Select appropriate mesh size for accurate wave simulation
3. Simulate sound waves across frequencies: 100 Hz to 5000 Hz
4. Compute sound pressure field
5. Calculate acoustic metrics:
   - **Transmission Loss (TL)**: Measures sound blocking capability
   - **Absorption Coefficient**: Measures sound energy absorption
   - **Noise Reduction Coefficient (NRC)**: Average absorption across 250-2000 Hz
6. Save results to CSV file

**Example:**
```python
python src/fem_simulation.py --input_dir data/geometries --output_file data/simulation_results.csv
```

### Step 3: Surrogate Model Development

Train machine learning models to predict acoustic performance without expensive FEM simulations.

**Key Steps:**
1. Import simulation dataset using Pandas
2. Split into inputs (geometry + material properties) and outputs (TL, NRC)
3. Train Random Forest Regressor
4. Use 80/20 train-test split
5. Evaluate using Mean Absolute Error (MAE)
6. Ensure MAE < 10% for acceptable accuracy
7. Save trained model

**Example:**
```python
python src/surrogate_model.py --data_file data/simulation_results.csv --model_file models/surrogate_model.pkl
```

### Step 4: Optimization

Use multi-objective optimization to find optimal panel designs.

**Key Steps:**
1. Set up NSGA-II algorithm using pymoo
2. Define objectives:
   - Maximize: TL, NRC, sustainability index
   - Minimize: cost, density
3. Run optimization using surrogate model
4. Generate Pareto front
5. Select top designs satisfying:
   - TL ≥ 35 dB (average)
   - NRC ≥ 0.85
   - Cost ≤ $20/m²
   - Sustainability index ≥ 0.75
6. Save selected geometries

**Example:**
```python
python src/optimization.py --model_file models/surrogate_model.pkl --output_file data/optimized_designs.csv
```

### Step 5: Results and Analysis

Visualize and analyze optimization results.

**Key Steps:**
1. Plot relationships between TL, NRC, cost, and sustainability
2. Identify trends between geometry parameters and performance
3. Highlight best designs (e.g., TL ≥ 35 dB and NRC ≥ 0.85)

**Example:**
```python
python src/visualization.py --results_file data/optimized_designs.csv --output_dir results/plots
```

## Material Properties

The framework supports four material types with the following properties:

| Material | Density (kg/m³) | Young's Modulus (Pa) | Poisson's Ratio | Cost ($/m²) | Sustainability Index |
|----------|----------------|---------------------|-----------------|-------------|---------------------|
| Foam | Variable | Variable | Variable | Variable | Variable |
| Cork | Variable | Variable | Variable | Variable | Variable |
| Recycled Plastic | Variable | Variable | Variable | Variable | Variable |
| Cardboard | Variable | Variable | Variable | Variable | Variable |

*Note: Actual values are defined in the material property database*

## Acoustic Metrics

### Transmission Loss (TL)
Measures the panel's ability to block sound transmission. Higher values indicate better sound isolation.

### Absorption Coefficient
Quantifies the fraction of incident sound energy absorbed by the panel. Ranges from 0 (no absorption) to 1 (complete absorption).

### Noise Reduction Coefficient (NRC)
Average absorption coefficient across the frequency range 250-2000 Hz. Provides a single-figure rating for acoustic absorption performance.

## Optimization Objectives

The multi-objective optimization balances:

- **Acoustic Performance**: Maximize TL and NRC
- **Sustainability**: Maximize sustainability index
- **Cost Efficiency**: Minimize material cost
- **Weight**: Minimize density

## Constraints

Designs must satisfy:

- **Geometric**: 50 ≤ A/t ≤ 300 (area-to-thickness ratio)
- **Cost**: Total cost ≤ maximum cost threshold
- **Sustainability**: Sustainability index ≥ baseline threshold (0.75)
- **Performance**: TL ≥ 35 dB, NRC ≥ 0.85
- **Size**: All geometries within 0.5 × 0.5 × 0.02 m bounds

## Output Files

- `simulation_results.csv`: Complete FEM simulation results with TL, absorption, and NRC for each design
- `optimized_designs.csv`: Pareto-optimal designs meeting all performance criteria
- `surrogate_model.pkl`: Trained Random Forest model for fast predictions
- Visualization plots: Pareto fronts, performance trends, and design comparisons

## Performance Targets

Top-performing designs must satisfy:

- **Transmission Loss**: ≥ 35 dB (average across frequency range)
- **Noise Reduction Coefficient**: ≥ 0.85
- **Cost**: ≤ $20/m²
- **Sustainability Index**: ≥ 0.75

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Shivank

## Acknowledgments

- FEniCS Project for finite element simulation capabilities
- PyMesh for geometry processing and meshing
- pymoo for multi-objective optimization algorithms

## References

- FEniCS Documentation: https://fenicsproject.org/
- PyMesh Documentation: https://pymesh.readthedocs.io/
- pymoo Documentation: https://pymoo.org/

## Troubleshooting

### Common Issues

**PyMesh Installation**
- PyMesh may require building from source. Follow the official installation guide.

**FEniCS Installation**
- FEniCS installation varies by platform. Use Docker containers for easier setup.

**Mesh Generation Failures**
- Ensure geometries are valid and watertight
- Adjust mesh size parameters if meshing fails

**Simulation Convergence**
- Reduce mesh size for better accuracy
- Check material property values are physically reasonable

**Model Training**
- Ensure sufficient training data (recommended: 1000+ samples)
- Check for missing values or outliers in the dataset

## Future Enhancements

- Support for additional materials
- Extended frequency range analysis
- Real-time optimization interface
- 3D visualization of optimized geometries
- Integration with CAD software
