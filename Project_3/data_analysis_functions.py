# Import built-in Python modules
from math import e

# Import libraries for data manipulation
import pandas as pd
import numpy as np

# Import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for geospatial data and maps
import folium
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
import branca

# Import libraries for display (e.g., in Jupyter Notebook)
from IPython.display import display

# Import libraries for statistics and modeling
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency
import pingouin as pg
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




def plot_comparison_between_original_and_corrected_dataset(df, df_refined):
  # Calculate the number of non-missing values for each column
  non_missing_original = df.notnull().sum()
  non_missing_refined = df_refined.notnull().sum()

  # Prepare data for the bar plot
  comparison_data = pd.DataFrame({
      'Columns': non_missing_original.index,
      'Original Dataset': non_missing_original.values,
      'Corrected Dataset': non_missing_refined.values
  }).melt(id_vars='Columns', var_name='Dataset', value_name='Non-Missing Count')


  # Create the bar plot
  plt.figure(figsize=(12, 6))
  ax = sns.barplot(
      data=comparison_data,
      x='Columns',
      y='Non-Missing Count',
      hue='Dataset',
      palette='pastel'
  )
  # Rotate the x-axis labels
  plt.xticks(rotation=45, ha='right')

  # Add count annotations to each bar (inside the bars)
  for p in ax.patches:
      height = p.get_height()  # Bar height (value of the bar)
      if height > 0:  # Only annotate bars with positive height
          ax.annotate(
              f'{int(height)}',  # Display the count as an integer
              (p.get_x() + p.get_width() / 2., height / 2),  # Position inside the bar, vertically centered
              ha='center', va='center', fontsize=10, color='black', rotation=90
          )

  # Add plot titles and labels
  plt.title('Comparison of Non-Missing Values by Column', fontsize=14)
  plt.ylabel('Number of Non-Missing Values', fontsize=12)
  plt.xlabel('Columns', fontsize=12)
  plt.legend(title='Dataset', loc='upper right')
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()  # Adjust layout to prevent overlap
  plt.show()

def plot_differences_between_original_and_corrected_dataset(df, df_refined):
  # Calculate the number of non-missing values for each column
  non_missing_original = df.notnull().sum()
  non_missing_refined = df_refined.notnull().sum()

  # Create a DataFrame for plotting
  diff_df = data_diff = (non_missing_refined - non_missing_original).reset_index()
  diff_df.columns = ['Columns', 'Difference']

  # Plot the data using Seaborn
  plt.figure(figsize=(10, 6))
  ax = sns.barplot(
      data=diff_df,
      x=diff_df.columns[0],
      y=diff_df.columns[1],
      hue = diff_df.columns[0],
      legend = False,
      palette=["blue" if x > 0 else "red" for x in diff_df['Difference']]  # Blue for positive, Red for negative
  )

  # Set the y-axis to be centered at 0
  ax.axhline(0, color='black',linewidth=1)

  # Add plot titles and labels
  plt.title('Differences Between Original and Corrected Datasets counts', fontsize=14)
  plt.ylabel('Difference (Original - Corrected)', fontsize=12)
  plt.xlabel('Columns', fontsize=12)

  # Rotate the x-axis labels
  plt.xticks(rotation=45, ha='right')

  # Display the values inside the bars
  for p in ax.patches:
      height = p.get_height()
      ax.text(
          p.get_x() + p.get_width() / 2,  # Position at the center of the bar
          height/2,  # Position at the top of the bar
          f'{int(height)}',  # Annotate with the number
          ha='center', va='center', fontsize=10, color='k', rotation=0
      )

  # Show the plot
  plt.tight_layout()  # Adjust the layout to avoid overlap
  plt.show()
  


def plot_univariate_variable_analysis(
    df: pd.DataFrame,
    variable_name: str,
    graph_type: str,
    variable_title_type: str = None,
    rotation: float = 90,
    is_splitted: bool = False,
) -> None:
    """
    Plots the distribution of a specified variable in a DataFrame based on the graph type.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variable_name (str): The name of the variable to plot.
        graph_type (str): The type of graph to plot (e.g., 'pie', 'heat_map', 'bars', 'hist', 'bars_log').
        variable_title_type (str, optional): An optional title type for the variable. Defaults to None.

    Raises:
        ValueError: If the graph_type is not recognized or if latitude/longitude columns are missing.
    """
    if variable_title_type is None:
        variable_title_type = graph_type

    graph_types = ['pie', 'heat_map', 'bars', 'hist', 'bars_log', 'boxplot']
    graph_type = graph_type.lower()

    graph_type_error = True
    for g_type in graph_types:
      if g_type in graph_type:
        graph_type_error = False

    if graph_type_error:
        raise ValueError('Error: wrong graph_type')

    if 'pie' in graph_type:
        if '%' in graph_type:
          threshold = float(graph_type.split('_')[-1][:-1])
        else:
          threshold = 2
        value_counts = df[variable_name].value_counts()  # Get counts without normalization
        total = value_counts.sum()  # Total count for percentage calculation

        # Group classes with less than 2% into "Others" (using raw counts)
        others_count = value_counts[value_counts / total * 100 < threshold]
        if not others_count.empty:
            # others_label = f"Others ({', '.join(others_count.index)})"
            others_label = "Others"
            value_counts = value_counts[value_counts / total * 100 >= threshold]
            value_counts[others_label] = others_count.sum()  # Add the "Others" category

        plt.pie(
            value_counts,
            labels=[f"{label} ({count/total*100:.1f}%)" for label, count in zip(value_counts.index, value_counts.values)],  # Percentage in labels
            autopct=lambda p: f'{int(p * total / 100)}'  # Count on wedges
        )
        plt.title(f"{variable_name} Distribution")
        plt.show()

    elif 'bars' in graph_type or 'bars_log' in graph_type:
        value_counts = df[variable_name].value_counts()

        tail = graph_type.split('_')[-1]
        if tail.isdigit():
          threshold = int(tail)
        else:
          threshold = 26

        # Group smaller classes into "others" if there are more than 26
        if len(value_counts) > threshold:
            top_categories = value_counts[:threshold-1]
            others_count = value_counts[threshold-1:].sum()
            value_counts = pd.concat([top_categories, pd.Series({f'others ({len(value_counts[threshold-1:])})': others_count})])

        if is_splitted:
            total = df.index.nunique()
        else:
            total = value_counts.sum()
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f"{variable_name} Distribution")

        # Add percentages inside the bars
        for p, percentage in zip(ax.patches, value_counts / total * 100):
            rotation = rotation if len(value_counts) > 10 else 0
            ax.annotate(
                f"{percentage:.1f}%",
                (p.get_x() + p.get_width() * 0.50, p.get_height() * 0.90),
                ha='center', va='center', fontsize=10, rotation=90, color='black'
            )

        plt.xlabel(variable_name)  # Set the X-axis label
        plt.ylabel('Count')  # Set the Y-axis label

        # Rotate X-axis labels if there are too many categories
        plt.xticks(rotation=rotation if len(value_counts) > 10 else 0)

        # Check if the graph should have a logarithmic scale
        if graph_type == 'bars_log':
            plt.yscale('log')  # Apply a logarithmic scale to the Y-axis

        # Get the current Y-axis ticks
        current_ticks = plt.yticks()[0]  # Retrieve the existing tick positions


        if graph_type =='bars_log':
            new_ticks = list(current_ticks) + [total]  # Add the custom tick
            new_ticks = sorted(new_ticks)  # Sort ticks to maintain proper tick order
        else:  # Handle the case for a cartesian (linear) scale
            new_ticks = sorted(list(current_ticks) + [total])  # Add the custom tick

        plt.gca().set_yticks(new_ticks)  # Update ticks on the Y-axis
        # Display the final plot
        plt.show()


    elif graph_type == 'hist':
        plt.figure(figsize=(10, 6))
        # sns.histplot(df[variable_name][df[variable_name] <= df[variable].quantile(0.99)], kde=True)
        sns.histplot(df[variable_name], kde=True, color='blue', bins=15)
        plt.title(f"{variable_name} Distribution")
        plt.xlabel(variable_name)
        plt.ylabel('Count')
        plt.show()

    elif graph_type == "boxplot":
        # sns.boxplot(y=df[variable_name][df[variable_name] <= df[variable_name].quantile(0.99)])
        sns.boxplot(y=df[variable_name])
        plt.title(f"{variable_name} Distribution (Boxplot)")
        plt.ylabel(variable_name)
        plt.show()

    elif graph_type == 'heat_map':
        # Check if the DataFrame contains the necessary geographic columns
        if 'geo_point_2d_a' in df.columns and 'geo_point_2d_b' in df.columns and variable_name in df.columns:
            # Group by the variable of interest (e.g., 'city', 'region') and count the occurrences in each sector
            sector_data = df.groupby([variable_name]).agg({
                'geo_point_2d_a': 'mean',
                'geo_point_2d_b': 'mean',
            }).reset_index()

            # Add a "population density" column based on the number of occurrences of the variable in each sector
            sector_data['counts'] = df.groupby(variable_name).size().values

            # Calculate the occurrence percentages
            total_counts = sector_data['counts'].sum()
            sector_data['percentage'] = (sector_data['counts'] / total_counts) * 100

            # Common zoom level used for city-level maps in Folium (and Leaflet)
            zoom_level = 12
            # Create a map centered around the average geographic points
            m = folium.Map(location=[sector_data['geo_point_2d_a'].mean(), sector_data['geo_point_2d_b'].mean()], zoom_start=zoom_level)

            # Normalize population densities for the heat map
            min_density = sector_data['counts'].min()
            max_density = sector_data['counts'].max()

            # Create the colormap with branca
            colormap = branca.colormap.LinearColormap(
                colors=['yellow', 'orange', 'red'],  # Choose a classic heatmap palette (yellow-orange-red)
                vmin=min_density, vmax=max_density
            ).to_step(10)  # Divide the colormap into 10 steps for the legend

            # Extract coordinates of the sectors
            coords = sector_data[['geo_point_2d_a', 'geo_point_2d_b']].values
            tree = cKDTree(coords)  # Create the k-d tree with the coordinates

            # Initialize the minimum distance to a very large number
            min_distance = float('inf')
            closest_pair = None

            # Find the closest pairs
            for i, point in enumerate(coords):
                # Find the closest neighbor from point i (excluding the point itself)
                distances, indices = tree.query(point, k=2)  # k=2 to get the closest neighbor and itself

                # The second closest will be the nearest (the first is the point itself)
                closest_distance = distances[1]
                closest_index = indices[1]

                # If the found distance is smaller, update the minimum distance and associated points
                if closest_distance < min_distance:
                    min_distance = closest_distance
                    closest_pair = (point, coords[closest_index])

            # Convert the minimum Euclidean distance to geodesic distance for more accuracy
            min_distance = geodesic(closest_pair[0], closest_pair[1]).meters

            # Limit the circle size to half the smallest distance between two points
            max_circle_radius = min_distance / 2

            # Average Earth radius in meters
            EARTH_RADIUS = 6371000

            # Calculate the size of a pixel in meters (at the given average latitude)
            lat_rad = np.radians(np.mean(sector_data['geo_point_2d_b']))
            meters_per_pixel = (2 * np.pi * EARTH_RADIUS) / (256 * (2 ** zoom_level)) * np.cos(lat_rad)

            max_circle_radius_in_pixels = max_circle_radius / meters_per_pixel

            # Add each sector as a colored circle based on population density
            for _, row in sector_data.iterrows():
                color = colormap(row['counts'])  # Use the colormap to get the color based on density

                # Calculate the size of the marker based on density
                # size = np.log(row['population_density'] + 1) * 5  # Logarithmic for a reasonable scale
                # size = max(10, min(size, max_circle_radius))  # Limit the size to max_circle_radius
                size = max_circle_radius_in_pixels

                # Add a circle to the map
                folium.CircleMarker(
                    location=[row['geo_point_2d_a'], row['geo_point_2d_b']],
                    radius=size,  # Circle size based on density and minimum distance
                    color=color,
                    fill=True,
                    fill_color=color,  # Fill with the same color
                    fill_opacity=0.7,
                    popup=f"{variable_name}: {row[variable_name]}<br>Trees: {row['counts']}<br>Percentage: {row['percentage']:.3}%",
                    # Allow the size to adjust based on zoom level
                    zoom_on_click=True
                ).add_to(m)

            # Add the colormap legend to the map
            colormap.caption = f"Trees per {variable_name}"
            colormap.add_to(m)

            # Force the legend to the bottom left corner using CSS style
            m.get_root().html.add_child(folium.Element("""
                <style>
                    .leaflet-control-colormap {
                        position: absolute !important;
                        bottom: 10px;
                        left: 10px;
                        z-index: 1000;
                    }
                </style>
            """))

            # Display the map
            display(m)
            m.save(f'./map_{variable_name}.html')
            return  # End the function

        else:
            raise ValueError("The DataFrame must contain 'geo_point_2d_a', 'geo_point_2d_b', and the sector variable to display a map.")



def plot_bivariate_variable_analysis(
    x_series: pd.Series,
    y_series: pd.Series,
    graph_type: str,
    title: str = None,
) -> None:
    """
    Plots the distribution of a specified variable using two Pandas Series and a specified graph type.

    Args:
        x_series (pd.Series): The first variable to plot (x-axis).
        y_series (pd.Series): The second variable to plot (y-axis).
        graph_type (str): The type of graph to plot (e.g., 'boxplot', 'scatter').
        variable_title_type (str, optional): An optional title for the graph. Defaults to None.

    Raises:
        ValueError: If the graph_type is not recognized.
    """

    # Define the supported graph types
    graph_types = ['boxplot', 'scatter']
    graph_type = graph_type.lower()  # Convert the graph type to lowercase for case insensitivity

    # Raise an error if the specified graph type is not supported
    if graph_type not in graph_types:
        raise ValueError(
            f"The graph type '{graph_type}' is not supported. "
            f"Accepted types: {', '.join(graph_types)}"
        )

    # Plot a boxplot if the specified graph type is "boxplot"
    if graph_type == "boxplot":
        boxplot_bivariate_analysis(x_series, y_series, title)

    # Plot a scatter plot if the specified graph type is "scatter"
    elif graph_type == "scatter":
        scatter_plot(x_series, y_series, title)

def boxplot_bivariate_analysis(
    x_series: pd.Series,
    y_series: pd.Series,
    title: str = None,
) -> None:
    """
    Plots a boxplot for the specified x_data and y_data.

    Args:
        x_series (pd.Series): The data for the x-axis.
        y_series (pd.Series): The data for the y-axis.
    """
    # Filter out NaN or infinite values from both series
    idx = np.isfinite(x_series) & np.isfinite(y_series)
    x_clean = x_series[idx]
    y_clean = y_series[idx]

    # Create the boxplot using Seaborn
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.boxplot(x=x_clean, y=y_clean)  # Generate the boxplot
    if title == None:
        plt.title(f'Boxplot {x_series.name} vs. {y_series.name}')  # Add the title
    else :
        plt.title(title)
    plt.xlabel(x_series.name)  # Set x-axis label
    plt.ylabel(y_series.name)  # Set y-axis label
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.show()  # Display the boxplot

def scatter_plot(
    x_series: pd.Series,
    y_series: pd.Series,
    title: str = None,
) -> None:
    """
    Plots a scatter plot for the specified x_data and y_data and overlays a linear regression line.

    Args:
        x_series (pd.Series): The data for the x-axis.
        y_series (pd.Series): The data for the y-axis.
    """
    # Filter out NaN or infinite values from both series
    idx = np.isfinite(x_series) & np.isfinite(y_series)
    x_clean = x_series[idx]
    y_clean = y_series[idx]

    # Calculate the linear regression line coefficients (slope and intercept)
    a, b = np.polyfit(x_clean, y_clean, 1)  # Fit a polynomial of degree 1 (linear regression)

    # Create the scatter plot with Seaborn
    plt.figure(figsize=(10, 6))  # Set the figure size
    sns.scatterplot(x=x_clean, y=y_clean, color='blue', alpha=0.6)  # Generate scatterplot

    # Plot the linear regression line
    x_fit = np.linspace(min(x_clean), max(x_clean), 100)  # Generate values for the x-axis of the line
    y_fit = a * x_fit + b  # Calculate corresponding y values for the line
    plt.plot(x_fit, y_fit, color='red', label=f'y = {format_coefficient(a)}x + {format_coefficient(b)}')  # Add the line

    # Customize the plot
    if title == None:
        plt.title("Scatter Plot with Linear Regression Line")  # Add the plot title
    else:
        plt.title(title)
    plt.xlabel(f'{x_series.name}')  # Set x-axis label
    plt.ylabel(f'{y_series.name}')  # Set y-axis label
    plt.legend()  # Display the legend
    plt.grid()  # Add a grid for better readability

    # Display the plot
    plt.show()


### Main bivariate indicators function
def compute_bivariate_stats_indicators(x_series: pd.Series, y_series: pd.Series) -> None:
    """
    Detects the types of series (categorical or quantitative) and determines the
    appropriate statistical analysis (ANOVA, Pearson Correlation, Chi-Squared Test).

    Args:
        x_series (pd.Series): The first series.
        y_series (pd.Series): The second series.

    Returns:
        None: Results are printed to the console.
    """
    # Step 1: Determine the type of each series
    if isinstance(x_series.dropna().iloc[0], str):
        x_type = 'categorical'
    else:
        x_type = 'quantitative'

    if isinstance(y_series.dropna().iloc[0], str):
        y_type = 'categorical'
    else:
        y_type = 'quantitative'

    # Step 2: Call the appropriate function based on the types
    if x_type == 'quantitative' and y_type == 'quantitative':
        # Both series are quantitative -> Pearson Correlation
        corr, p_value = perform_pearson_correlation(x_series, y_series)
        result = corr, p_value

    elif x_type == 'categorical' and y_type == 'categorical':
        # Both series are categorical -> Chi-Squared Test
        contingency_table, v_cramer, p, chi2, dof, expected = perform_chi2_test(x_series, y_series)
        result = contingency_table, v_cramer, p, chi2, dof, expected

    elif (x_type == 'categorical' and y_type == 'quantitative') or (
        x_type == 'quantitative' and y_type == 'categorical'
    ):
        # One series is categorical and the other is quantitative -> ANOVA
        anova_table, eta_squared, p_value = perform_anova(x_series, y_series)
        result = anova_table, eta_squared, p_value

    else:
        raise ValueError("Invalid combination of variable types.")

    return result

### Pearson Correlation Function
def perform_pearson_correlation(x_series: pd.Series, y_series: pd.Series) -> tuple:
    """
    Computes the Pearson correlation coefficient for two quantitative variables.

    Args:
        x_series (pd.Series): The first quantitative series.
        y_series (pd.Series): The second quantitative series.

    Returns:
        tuple: A tuple containing:
            - the correlation coefficient (corr),
            - the p-value (p_value).
    """
    # Remove NaN values from both series
    valid_idx = x_series.notna() & y_series.notna()
    x_clean = x_series[valid_idx]
    y_clean = y_series[valid_idx]

    # Compute Pearson correlation
    corr, p_value = pearsonr(x_clean, y_clean)

    return corr, p_value
  
def perform_anova(x_series: pd.Series, y_series: pd.Series) -> tuple:
    """
    Performs an ANOVA (analysis of variance) test using Pingouin
    and returns results.

    Args:
        x_series (pd.Series): The categorical series.
        y_series (pd.Series): The quantitative series.

    Returns:
        tuple: A tuple containing:
            - a DataFrame of the ANOVA results (anova_table),
            - Eta-squared (eta_squared),
            - the p-value (p_value).
    """
    # Determine if x_series is categorical
    if x_series.dtype == 'O' or isinstance(x_series.iloc[0], str):
        category_column = 'x_series'
        dependent_column = 'y_series'
    else:
        category_column = 'y_series'
        dependent_column = 'x_series'

    # Create a temporary DataFrame for the ANOVA analysis
    df_temp = pd.DataFrame({category_column: x_series, dependent_column: y_series})

    # Exclure missing or empty values
    df_temp = df_temp[df_temp[category_column] != ""].dropna()

    # unidirectional ANOVA with Pingouin
    anova_table = pg.anova(data=df_temp, dv=dependent_column, between=category_column, detailed=True)

    # Compute Eta-squared
    eta_squared = anova_table["np2"].iloc[0]  # Pingouin fournit directement l'eta-carre partiel (np2)

    # Extraire la p-value
    p_value = anova_table["p-unc"].iloc[0]

    return anova_table, eta_squared, p_value

### Chi-Squared Test Function
def perform_chi2_test(x_series: pd.Series, y_series: pd.Series, adjusted=True) -> tuple:
    """
    Performs Chi-Squared test and computes Cramer's V (optionally adjusted).

    Args:
        x_series (pd.Series): First categorical variable.
        y_series (pd.Series): Second categorical variable.
        adjusted (bool): If True (default), returns adjusted Cramer's V. 
                         False returns standard V.

    Returns:
        tuple: contingency_table, v_retour, p, chi2, dof, expected
    """
    # Create contingency table
    contingency_table = pd.crosstab(x_series, y_series)
    
    # Compute Chi-Squared stats
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Prepare parameters
    n = contingency_table.sum().sum()
    min_dof = min(contingency_table.shape[0]-1, contingency_table.shape[1]-1)
    k = min(contingency_table.shape)  # min(r, c)
    
    # Base V (standard)
    v_standard = np.sqrt(chi2 / (n * min_dof)) if min_dof > 0 else 0
    
    # Adjusted V 
    if adjusted:
        if k <= 1:
            v_adj = 0.0
        else:
            v_max = np.sqrt(min_dof / (k - 1))
            v_adj = v_standard / v_max if v_max != 0 else 0.0
        v_cramer_adj = v_adj
    else:
        v_cramer_adj = v_standard
    
    return contingency_table, v_cramer_adj, p, chi2, dof, expected
    
    
def correlation_matrix_with_p_values(df: pd.DataFrame, rotation=90):
    """
    Computes a correlation matrix with statistical significance annotations.
    
    - ✔ : p < 0.05 (significant correlation)
    - ✖ : p ≥ 0.05 (non-significant correlation)
    
    Based on the discussion here: https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
    """

    # Compute the correlation matrix (rho)
    rho = df.corr()

    # Compute the p-value matrix (pval) using pearsonr and adjust for diagonal values
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)

    # Add annotations (✔ for p < 0.05, ✖ otherwise) to each cell
    p = pval.map(lambda x: '\n✔' if x < 0.05 else '\n✖')

    # Combine correlation values with significance annotations
    annotated_matrix = rho.round(2).astype(str) + p

    # Plot the heatmap with annotations
    plt.figure(figsize=(12, 10))
    sns.heatmap(rho, annot=annotated_matrix, fmt="", cmap='coolwarm', cbar=True, square=True)
    plt.xticks(rotation=rotation)
    plt.title('Correlation Matrix with Significance (✔ = p < 0.05, ✖ = p ≥ 0.05)', fontsize=16)
    plt.show()
    
def eta_squared_with_p_heatmap(df, dependent_var, rotation=90):
    """
    Plots a heatmap of Eta-squared coefficients with p-value annotations.
    
    Handles both cases:
    - dependent_var is quantitative, and other variables are categorical.
    - dependent_var is categorical, and other variables are quantitative.
    
    Explodes list-like categorical columns automatically for proper ANOVA computation.

    Args:
    - df : DataFrame containing the dataset.
    - dependent_var : Name of the dependent variable.
    - rotation : Rotation angle for variable labels on the X-axis.
    """
    # Helper function to detect if a column is categorical
    def is_categorical(col):
        """A column is categorical if it contains only strings or lists of strings."""
        return (
            df[col].apply(lambda x: isinstance(x, (str, list))).all()
        )

    # Helper to explode list-like columns
    def explode_if_needed(df, categorical_col):
        """Explodes a column if it contains list-like elements."""
        if df[categorical_col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(categorical_col)
        return df

    # Determine if `dependent_var` is categorical or quantitative
    if is_categorical(dependent_var):
        dependent_type = "categorical"
        independent_vars = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        dependent_type = "quantitative"
        independent_vars = df.select_dtypes(include=["category", "object"]).columns.tolist()

    # Ensure the independent variables exclude the dependent variable
    independent_vars = [col for col in independent_vars if col != dependent_var]

    # Containers for results
    eta_squareds = []
    p_values = []

    # Iterate over independent variables
    for col in independent_vars:
        # Copy the dataframe for processing
        temp_df = df.copy()

        # If dependent_var is categorical, check and explode if needed
        if dependent_type == "categorical":
            temp_df = explode_if_needed(temp_df, dependent_var)

        # If the current independent variable is categorical, check and explode if needed
        elif is_categorical(col):
            temp_df = explode_if_needed(temp_df, col)
            # Convert to 'category' type for ANOVA compatibility
            if temp_df[col].dtype != "category":
                temp_df[col] = temp_df[col].astype("category")

        # Perform ANOVA
        try:
            if dependent_type == "quantitative":
                # Quantitative dependent variable -> Categorical independent
                anova = pg.anova(dv=dependent_var, between=col, data=temp_df, detailed=True)
            else:
                # Categorical dependent variable -> Quantitative independent
                anova = pg.anova(dv=col, between=dependent_var, data=temp_df, detailed=True)

            # Extract eta-squared and p-value
            eta_squared = anova.loc[0, "np2"]
            p_value = anova.loc[0, "p-unc"]

        except Exception as e:
            # Skip problematic columns if ANOVA cannot be performed
            print(f"Skipping column {col} due to error: {e}")
            eta_squared, p_value = None, None

        eta_squareds.append(eta_squared)
        p_values.append(p_value)

    # Create DataFrame for the heatmap
    eta_sq_df = pd.DataFrame([eta_squareds], columns=independent_vars)
    p_value_df = pd.DataFrame([p_values], columns=independent_vars)

    # Add significance annotations
    annotations = p_value_df.map(lambda x: "\n✔" if x < 0.05 else "\n✖" if x is not None else "\n-")
    annotated_matrix = eta_sq_df.round(2).astype(str).replace("nan", "") + annotations

    # Determine labels based on the nature of `dependent_var`
    if dependent_type == "quantitative":
        ylabel = "Quantitative Variable"
        xlabel = "Categorical Variables"
    else:
        ylabel = "Categorical Variable"
        xlabel = "Quantitative Variables"

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(eta_sq_df, annot=annotated_matrix, fmt="", cmap="coolwarm", cbar=True, square=False)
    plt.xticks(rotation=rotation, fontsize=10)
    plt.yticks([0], [dependent_var], rotation=0, fontsize=12)
    plt.title("η² Heatmap with Significance (✔ = p < 0.05, ✖ = p ≥ 0.05, - = Error)", fontsize=16)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
    
def cramers_v_heatmap(df: pd.DataFrame, dependent_var: str, rotation: int = 90) -> None:
    """
    Displays a heatmap of adjusted Cramer's V between a dependent column and other categorical features,
    with sequential list explosion handling.
    
    Args:
        df: Input DataFrame containing categorical data
        dependent_var: Target column name for association analysis
        rotation: X-axis label rotation (degrees)
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import chi2_contingency

    # Duplicate columns check
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns detected: {dup_cols}")

    # Data preparation
    df_clean = df[[dependent_var] + [c for c in df.columns if c != dependent_var]].copy()
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].astype(str)  # Convert numeric columns to categorical

    # Results storage
    results = {
        'adj_cramers_v': [],
        'p_value': [],
        'k_value': [],
        'features': []
    }

    # Feature analysis loop
    for feature in [c for c in df_clean.columns if c != dependent_var]:
        try:
            # Sequential explosion with index alignment
            target_exploded = df_clean[[dependent_var]].explode(dependent_var).dropna()
            feature_exploded = df_clean[[feature]].explode(feature).dropna()
            
            merged_df = target_exploded.merge(
                feature_exploded,
                left_index=True,
                right_index=True,
                how='inner'
            ).dropna()

            if merged_df.empty:
                print(f"Skipped {feature}: Not enough valid data after explosion")
                continue

            # Contingency table creation
            contingency_table = pd.crosstab(merged_df[dependent_var], merged_df[feature])
            
            # Statistical parameters
            n_obs = contingency_table.sum().sum()
            n_rows, n_cols = contingency_table.shape
            min_degrees = min(n_rows - 1, n_cols - 1)
            min_dim = min(n_rows, n_cols)
            
            # Chi2 test results
            chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
            
            # Adjusted Cramer's V calculation
            if min_degrees == 0 or min_dim <= 1:
                adj_v = 0.0
            else:
                std_v = np.sqrt(chi2_stat / (n_obs * min_degrees))
                max_v = np.sqrt(min_degrees / (min_dim - 1))
                adj_v = std_v / max_v

            # Store results
            results['adj_cramers_v'].append(adj_v)
            results['p_value'].append(p_val)
            results['k_value'].append(min_dim)
            results['features'].append(feature)

        except Exception as e:
            print(f"Error processing {feature}: {str(e)}")
            continue

    # Heatmap data preparation
    v_matrix = pd.DataFrame([results['adj_cramers_v']], columns=results['features'])
    k_matrix = pd.DataFrame([results['k_value']], columns=results['features'])
    p_matrix = pd.DataFrame([results['p_value']], columns=results['features'])

    # Annotation formatting
    annotations = (
        v_matrix.round(2).astype(str) + 
        "\n(k=" + k_matrix.astype(str) + ")" +
        p_matrix.map(lambda p: " ✔" if p < 0.05 else " ✖")
    )

    # Visualization settings
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        v_matrix,
        annot=annotations,
        fmt="",
        cmap="coolwarm",
        cbar_kws={'label': 'Adjusted Cramer\'s V'},
        linewidths=0.5,
        annot_kws={"size": 10, "color": "white"}
    )
    
    plt.title(f"Cramer's V Heatmap Association Strength: {dependent_var} vs Features\n(✔ = p < 0.05, ✖ = p ≥ 0.05)", pad=15)
    plt.xticks(rotation=rotation)
    plt.yticks([])
    plt.xlabel("Features")
    plt.show()
    
def pca_analysis(
    df: pd.DataFrame,
    n_components: int = 2,
    normalize: bool = True,
    rotation: int = 90,
    figsize: tuple = (14, 8)
) -> None:
    """
    Performs PCA analysis with variance explanation and annotated component loadings.
    
    Args:
        df: Quantitative variables DataFrame
        n_components: Number of principal components to keep
        normalize: Whether to standardize variables (default True)
        rotation: X-axis label rotation
        figsize: Output figure dimensions
    """
    # Check for duplicates
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns detected: {dup_cols}")

    df_clean = df.dropna()

    # Standardization
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
    else:
        X = df_clean.values

    # PCA computation
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    # Build components dataframe
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    components_df = pd.DataFrame(
        loadings,
        index=df_clean.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Create visualization grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.5])

    # Variance explanation plot
    ax1 = fig.add_subplot(gs[0])
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    ax1.plot(range(1, n_components+1), cum_var, 'o-', color='#2e5e87')
    ax1.set_ylabel('Cumulative Explained Variance', fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(range(1, n_components+1))

    # Annotate variance percentage
    for i, (ev, cv) in enumerate(zip(explained_var, cum_var)):
        ax1.text(i+1, cv + 0.03, 
                f"{cv*100:.1f}%\n(cumul.)\n——\n{ev*100:.1f}%", 
                ha='center', 
                fontsize=8,
                color='#174a6e')

    # Loadings heatmap
    ax2 = fig.add_subplot(gs[1])
    heatmap = sns.heatmap(
        components_df.T,
        annot=components_df.T.round(2),
        fmt="",
        cmap='coolwarm',
        cbar_kws={'label': 'Loading Value'},
        linewidths=0.5,
        ax=ax2
    )

    # Formatting
    heatmap.set_yticklabels(
        [f"PC{i+1}\n({ev*100:.1f}%)" for i, ev in enumerate(explained_var)],
        rotation=0
    )
    plt.xticks(rotation=rotation)
    plt.title(f"PCA Component Loadings (Normalized={normalize})", pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()



def compute_distances(df, tree_index):
    """
    Computes distances between a specific tree and all other trees in the DataFrame.

    Args:
        df: DataFrame containing tree data with 'geo_point_2d_a' (longitude) and 'geo_point_2d_b' (latitude) columns.
        tree_index: Index of the specific tree in the DataFrame.

    Returns:
        A pandas Series containing distances to all other trees from the specified tree.
        Returns None if the tree_index is out of range or if necessary columns are missing.
    """

    # Safety checks
    if not all(col in df.columns for col in ['geo_point_2d_a', 'geo_point_2d_b']):
        print("Error: 'geo_point_2d_a' or 'geo_point_2d_b' columns not found in DataFrame.")
        return None

    if tree_index not in df.index:
        print(f"Error: tree_index {tree_index} is out of range.")
        return None

    # Get coordinates of the chosen tree
    tree_coords = (df.loc[tree_index, 'geo_point_2d_a'], df.loc[tree_index, 'geo_point_2d_b'])

    # Calculate distances to all other trees
    distances = []
    for index, row in df.iterrows():
        other_tree_coords = (row['geo_point_2d_a'], row['geo_point_2d_b'])  # Assuming latitude, longitude order
        distance = geodesic(tree_coords, other_tree_coords).meters
        distances.append(distance)

    return pd.Series(distances, index=df.index)  # Use original index
    
    
def format_coefficient(value, limit=5):
    threshold = 0.01
    digits_after_coma = 2
    while abs(value) < threshold:
        threshold /= 10
        digits_after_coma += 1
    if digits_after_coma > limit:
        digits_after_coma = 0
    # return f'{round(value,digits_after_coma):.{digits_after_coma}f}'
    return f'{round(value,digits_after_coma)}'
    
    
# to drop categories inside of my categorical variable each time the count < 10
def drop_infrequent_categories(df, column, threshold=10):
    """Drops categories from a categorical variable if their count is below a threshold."""
    value_counts = df[column].value_counts()
    categories_to_keep = value_counts[value_counts >= threshold].index
    df[column] = df[column].astype('string').apply(lambda x: x if x in categories_to_keep else 'Other')
    return df