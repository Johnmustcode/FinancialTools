import pandas as pd
import numpy as np
from scipy.stats import gmean

# Calculates the composite annual return of a specified column in a DataFrame and return the calculated return.
def calculate_composite_annual_return(target_df, monthly_return_data_column, output_column):
    """
    Parameters:
    - target_df (pandas.DataFrame): The DataFrame containing the original return data.
    - original_column (str): The name of the column with the original returns data, e.g., monthly returns.
    - target_column (str): The name of the new column where the calculated composite annual returns will be stored.

    Returns:
    - pandas.DataFrame: The updated DataFrame that includes the new column with the composite annual returns.

    Note:
    - This function assumes the original column data represents monthly returns and converts these into an annual composite return rate.
    
    """

    # Copy the DataFrame to avoid modifying the original data
    dfr = target_df.copy()

    # Add 1 to the original returns to facilitate geometric mean calculation
    dfr[output_column] = dfr[monthly_return_data_column] + 1

    # Calculate the geometric mean, which is used to find the average monthly growth rate
    dfr[output_column] = gmean(dfr[output_column])

    # Convert the average monthly growth rate to an annual growth rate by raising it to the power of 12
    dfr[output_column] = np.power(dfr[output_column], 12)

    # Subtract 1 from the annual growth rate to convert it back to standard return format
    dfr[output_column] = dfr[output_column] - 1

    return dfr


# Calculates the Sharpe Ratio for a specified returns column in a DataFrame and stores the result in a new column.
def calculate_sharpe_ratio(target_df, returns_column, risk_free_rate, output_column):
    """
    
    Parameters:
    - target_df (pandas.DataFrame): The DataFrame containing the return data.
    - returns_column (str): The name of the column with the returns data from which the Sharpe Ratio will be calculated.
    - risk_free_rate (float): The risk-free rate of return, expressed as a decimal. For example, use 0.01 for 1%.
    - output_column (str): The name of the new column where the calculated Sharpe Ratios will be stored.

    Returns:
    - pandas.DataFrame: The updated DataFrame that includes the new column with the Sharpe Ratios.

    Note:
    - The Sharpe Ratio is calculated as (mean return - risk-free rate) / standard deviation of the return.
    - This function assumes the returns are already expressed in excess of the risk-free rate if you want a pure measure of volatility. Otherwise, adjust the returns within the function as needed.
    """

    # Copy the DataFrame to avoid modifying the original data
    dfr = target_df.copy()

    # Calculate excess returns by subtracting the risk-free rate from the returns
    dfr['excess_returns'] = dfr[returns_column] - risk_free_rate

    # Calculate the mean of the excess returns
    mean_excess_return = dfr['excess_returns'].mean()

    # Calculate the standard deviation of the returns
    std_deviation = dfr[returns_column].std()

    # Calculate the Sharpe Ratio and store it in the output column
    # Note: We assume the risk-free rate is already annualized and matches the period of the returns
    dfr[output_column] = mean_excess_return / std_deviation

    # Return the updated DataFrame with the new Sharpe Ratio column
    return dfr