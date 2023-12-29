# Panel Data Analysis

## Subtitle: Panel Data Analysis on Densely Inhabited District Population Density and Relative Total Factor Productivity

### Detailed Description
This project aims to perform a comprehensive analysis of panel data. The key steps include:
- First, conducting pooled Ordinary Least Squares (pooled OLS) estimation.
- Next, performing analyses using Fixed Effects and Random Effects models.
- Finally, executing the Hausman test to evaluate model specifications.

### How to Run
To run this project, Docker is used to ensure a consistent environment and reproducibility. Please follow these steps:

1. **Building the Docker Image**:
   - This step compiles the Docker image with all necessary dependencies and project files.
   - Run the following command in the terminal:
     ```bash
     docker build -t panel .
     ```

2. **Running the Docker Container**:
   - After building the image, you can run the container which executes the Python script.
   - Use this command to run the container:
     ```bash
     docker run panel
     ```