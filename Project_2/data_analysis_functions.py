import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import branca
from IPython.display import display
from scipy.spatial import cKDTree
from geopy.distance import geodesic
import numpy as np


def plot_univariate_variable_analysis(
    df: pd.DataFrame, 
    variable_name: str, 
    graph_type: str, 
    variable_title_type: str = None,
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

    if graph_type not in graph_types:
        raise ValueError('Error: wrong graph_type')

    if graph_type == 'pie':
        value_counts = df[variable_name].value_counts()  # Get counts without normalization
        total = value_counts.sum()  # Total count for percentage calculation

        # Group classes with less than 2% into "Others" (using raw counts)
        others_count = value_counts[value_counts / total * 100 < 2]
        if not others_count.empty:
            others_label = f"Others ({', '.join(others_count.index)})"
            value_counts = value_counts[value_counts / total * 100 >= 2]
            value_counts[others_label] = others_count.sum()  # Add the "Others" category

        plt.pie(
            value_counts,
            labels=[f"{label} ({count/total*100:.1f}%)" for label, count in zip(value_counts.index, value_counts.values)],  # Percentage in labels
            autopct=lambda p: f'{int(p * total / 100)}'  # Count on wedges
        )
        plt.title(f"{variable_name} Distribution")

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

    elif graph_type in ['bars', 'bars_log']:
        value_counts = df[variable_name].value_counts()

        # Group smaller classes into "others" if there are more than 26
        if len(value_counts) > 26:
            top_categories = value_counts[:25]
            others_count = value_counts[25:].sum()
            value_counts = pd.concat([top_categories, pd.Series({f'others ({len(value_counts[25:])})': others_count})])

        total = value_counts.sum()
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f"{variable_name} Distribution")
        
        # Add percentages inside the bars
        for p, percentage in zip(ax.patches, value_counts / total * 100):
            rotation = 90 if len(value_counts) > 10 else 0
            ax.annotate(
                f"{percentage:.1f}%",
                (p.get_x() + p.get_width() * 0.50, p.get_height() * 0.90),
                ha='center', va='center', fontsize=10, rotation=rotation, color='black'
            )
        
        plt.xlabel(variable_name)
        plt.ylabel('Count')
        if graph_type == 'bars_log':
            plt.yscale('log')
        plt.xticks(rotation=90 if len(value_counts) > 10 else 0)  # Rotate labels if too many
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




def plot_bivariate_variable_analysis(
    df: pd.DataFrame, 
    variable1_name: str, 
    variable2_name: str,
    graph_type: str, 
    variable_title_type: str = None,
) -> None:
    """
    Plots the distribution of a specified variable in a DataFrame based on the graph type.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variable1_name (str): The name of the first variable to plot. (x)
        variable2_name (str): The name of the second variable to plot. (y)
        graph_type (str): The type of graph to plot (e.g., 'pie', 'heat_map', 'bars', 'hist', 'bars_log').
        variable_title_type (str, optional): An optional title type for the variable. Defaults to None.

    Raises:
        ValueError: If the graph_type is not recognized or if latitude/longitude columns are missing.
    """
    if variable_title_type is None:
        variable_title_type = graph_type

    graph_types = ['boxplot']
    graph_type = graph_type.lower()

    if graph_type == "boxplot":
      plt.figure(figsize=(8, 6))
      sns.boxplot(x=variable1_name, y=variable2_name, data=df)
      plt.title(f'Boxplot {variable1_name} vs. {variable2_name}')
      plt.xlabel(variable1_name)
      plt.ylabel(variable2_name)
      plt.show()

def compute_bivariate_stats_indicators(    
    df: pd.DataFrame, 
    variable1_name: str, 
    variable2_name: str,
) -> None:
    """
        Computes statistic indicator in order to determine possible correspondance between two variables.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            variable1_name (str): The name of the first variable. (x)
            variable2_name (str): The name of the second variable. (y)
    """
    import statsmodels.formula.api as sm
    from statsmodels.stats.anova import anova_lm
    import pandas as pd
    from scipy.stats import pearsonr


    variables_types = {}
    variable_names = [variable1_name, variable2_name]
    for variable_name in variable_names:
      if type(df[df[variable_name].notna()][variable_name].values[0]) == str:
        variables_types[variable_name] = 'categorical'
      else:
        variables_types[variable_name] = 'quantitative'

    if ('quantitative' in variables_types.values()) and ('categorical' in variables_types.values()): # ANOVA analysis
      print("Both quantitative and categorical variables : ANOVA analysis")
        # Build the formula string
      if variables_types[variable1_name] == 'quantitative' and variables_types[variable2_name] == 'categorical':
          formula_string = f'{variable1_name} ~ C({variable2_name})' 
      elif variables_types[variable1_name] == 'categorical' and variables_types[variable2_name] == 'quantitative':
          formula_string = f'{variable2_name} ~ C({variable1_name})'

      # Fit the ANOVA model
      model = sm.ols(formula_string, data=df).fit()
      anova_table = anova_lm(model, typ=2)

      # Extract eta_squared
      eta_squared = anova_table['sum_sq'][0] / sum(anova_table['sum_sq'])

      result = eta_squared, anova_table

    elif 'quantitative' in variables_types.values(): # Pearson correlation analysis
      print("Both quantitative variables : Pearson correlation analysis")

      # Remove rows with NaN values in either 'remarquable', 'circonference_cm', or 'hauteur_m'
      data_for_correlation = df.dropna(subset=[variable1_name, variable2_name])

      # Calculate Pearson correlation coefficient between 'remarquable' and 'circonference_cm'
      correlation_circonference, p_value_circonference = pearsonr(
          data_for_correlation[variable1_name], data_for_correlation[variable2_name]
      )

      result = correlation_circonference, p_value_circonference


    else: # 2 categorical variables, khi2 to add
      pass

    return result


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
