# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
import os #to work with files

# Visualization import section
from Visualization.visualization import doubleLinePlot, DoubleBoxPlot, Histogram, ScatterPlot, interpolation_subplot

# S&F import
from .smoothing_and_filtering_functions import Remove_Outliers, Smoothing, TimeSeriesOOP, Converter


def import_dset(data_obj):
    """Creates a current dataframe either from the original data object,
       or from the Filtered dataset, depending on which one is 'fresher'.
       Also, a datetime conversion is attempted in case of a time-
       series dataset.

    :param data_obj: DataObject instance
    :type data_obj: __main__.DataObject
    :return: pandas dataframe object
    :rtype: pandas.core.frame.DataFrame
    """
    try:
        # Read Filtered dataset if it exists
        a = pd.read_csv('Smoothing_and_Filtering//Filtered Dataset.csv', index_col = None)
        
        # Compare if Filtered dataset and the original dataframe are equal
        # If they are not equal then the .csv is taken
        if a.equals(data_obj.df) == False:
            current_df1 = a
            current_df = Converter.dateTime_converter(current_df1)
        # If equal then the original dataframe is taken
        else:
            current_df = data_obj.df
            current_df = Converter.dateTime_converter(current_df)
    # If there is no Filtered dataset, the original is taken as the current
    except:
        current_df = data_obj.df
        current_df = Converter.dateTime_converter(current_df)
        
    return current_df


def main(data_obj):
    """Smoothing and Filtering main

    :param data_obj: DataObject instance
    :type data_obj: __main__.DataObject
    """
    
    # Header
    st.header("Smoothing and filtering")
    # Instruction
    with st.expander(label="How to use", expanded=True):
        st.info("""
                On this page you will see three dataframes.
                \n1) Original dataframe as it looked like when you uploaded it
                \n2) Current dataframe is the one the is used at the moment
                \n3) Resulting dataframe is the one you will get after applying the changes. If will become a current dataframe then.
                \n
                \n To apply changes in-between intermediate steps do not forget to press 'Save intermediate results' button!
                \n
                \n If you mess up your dataset during 'Smoothing and Filtering', you can always reset it using the button on the (left) sidebar.
                """)
        st.warning("**No matter if you did any changes or not -- do not forget to press 'Finalize' button after you are done with 'Smoothing and filtering'!**")    

    # A button to circumvent loading the dataset used in the last session
    if st.sidebar.button("Reset dataframe to the initial one"):
        data_obj.df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

    # Create the current dataframe
    current_df = import_dset(data_obj)
    
    # Overview of the dataframes
    col1, col2, col3 = st.columns(3)

    # Original dataframe display
    with col1:
        st.subheader('Original dataframe')
        st.dataframe(data_obj.df)
        st.write(data_obj.df.shape)

    # Horizontal styling of radio buttons
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)
    
    # Main data preparation method radio selector
    dp_method = st.radio(label = 'Filtering Method', options = ['Remove outliers','Smoothing','Interpolation'])
    
    # Selected 'Remove outliers'
    if dp_method == 'Remove outliers':

        # Specifying remove outliers submethods
        rmo_radio = st.radio(label = 'Remove outliers method',
                             options = ['Std','Q','Z'])

        # Standard deviation method selected
        if rmo_radio == 'Std':

            # Standard deviation method main
            with st.container():
                st.subheader('Remove outliers using standard deviation')

                # Settings layout
                cc1, cc2, cc3, cc4 = st.columns(4)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    rm_outlier = Remove_Outliers.removeOutlier(current_df, selected_column, std_coeff)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])
                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-rm_outlier.shape[0]} rows will be removed.')
                   
                
                # Plotting
                if scatter_plot:
                    ScatterPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column, scatter_column2)
                if bp:
                    DoubleBoxPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(rm_outlier.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate remove outlier results (std)"):
                    current_df = rm_outlier.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

        # Quantile range method selected
        if rmo_radio == 'Q':

            # Quantile range method main
            with st.container():
                st.subheader('Remove outliers using quantiles')

                # Settings layout
                cc1, cc2, cc3, cc4 = st.columns(4)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    q_values = st.slider('Select a range of quantiles',
                                        0.00, 1.00, (0.25, 0.75))
                    selected_column = st.selectbox("Select a column:", columns_list)
                    q_outlier = Remove_Outliers.removeOutlier_q(current_df, selected_column, q_values[0], q_values[1])
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])
                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-q_outlier.shape[0]} rows will be removed.')
                        
                # Plotting
                if scatter_plot:
                    ScatterPlot(data_obj.df, q_outlier.reset_index(drop=True), selected_column, scatter_column2)
                if bp:
                    DoubleBoxPlot(data_obj.df, q_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(q_outlier.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate remove outlier results (Q)"):
                    current_df = q_outlier.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

        # Z-score method selected
        if rmo_radio == 'Z':

            # Z-score method main
            with st.container():
                st.subheader('Remove outliers using Z-score')

                # Settings layout
                cc1, cc2, cc3, cc4 = st.columns(4)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient: ", 0.0, 3.1, 2.0, 0.1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    z_outlier = Remove_Outliers.removeOutlier_z(current_df, selected_column, std_coeff)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])
                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-z_outlier.shape[0]} rows will be removed.')
                        
                # Plotting
                if scatter_plot:
                    ScatterPlot(data_obj.df, z_outlier.reset_index(drop=True), selected_column, scatter_column2)
                if bp:
                    DoubleBoxPlot(data_obj.df, z_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(z_outlier.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate remove outlier results (Z)"):
                    current_df = z_outlier.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)


    # Selected smoothing
    if dp_method == 'Smoothing':

        # Specifying smoothing submethods
        smooth_radio = st.radio(label = 'Smoothing',
                             options = ['Median filter','Moving average','Savitzky Golay'])

        # Median filter method selected
        if smooth_radio == 'Median filter':
            
            # Median filter method main
            with st.container():
                st.subheader('Median filter')

                # Settings layout
                cc1, cc2, cc3 = st.columns(3)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 3, 7, 5, 2)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    median_filt = Smoothing.median_filter(current_df, selected_column, filter_len)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                
                # Plotting
                if plot_basic:
                    doubleLinePlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)
                if bp:
                    DoubleBoxPlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(median_filt.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate smoothing results (median filter)"):
                    current_df = median_filt.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

        # Moving average method selected
        if smooth_radio == 'Moving average':

            # Moving average method main
            with st.container():
                st.subheader('Moving average')

                # Settings layout
                cc1, cc2, cc3 = st.columns(3)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 1, 10, 5, 1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    # median_filt = median_filter(current_df, selected_column, filter_len)
                    moving_ave = Smoothing.moving_average(current_df, selected_column, filter_len)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-moving_ave.shape[0]} rows will be removed.')
                
                # Plotting
                if plot_basic:
                    doubleLinePlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)
                if bp:
                    DoubleBoxPlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(moving_ave.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate smoothing results (moving average)"):
                    current_df = moving_ave.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

        # Savitzky Golay method selected
        if smooth_radio == 'Savitzky Golay':

            # Savitzky Golay method selected
            with st.container():
                st.subheader('Savitzky Golay')

                # Settings layout
                cc1, cc2, cc3 = st.columns(3)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 1, 11    , 5, 2)
                    order = st.slider('Polynomial order', 1, 3, 2, 1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    sg = Smoothing.savitzky_golay(current_df, selected_column, filter_len, order)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-sg.shape[0]} rows will be removed.')
                
                # Plotting
                if plot_basic:
                    doubleLinePlot(data_obj.df, sg.reset_index(drop=True), selected_column)
                if bp:
                    DoubleBoxPlot(data_obj.df, sg.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(sg.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate smoothing results (Savitzky Golay)"):
                    current_df = sg.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

    
    # Selected 'Interpolation'
    if dp_method == 'Interpolation':

        # Specifying interpolation submethods
        interpolation_radio = st.radio(label = 'Interpolation',
                             options = ['Linear','Cubic', 'Forward Fill', 'Backward Fill'])
    
        # Linear method selected
        if interpolation_radio == 'Linear':
            
            # Linear interpolation method selected
            with st.container():
                st.subheader('Linear interpolation')

                # Settings layout
                cc1, cc2, cc3 = st.columns(3)
            
                # Input settings
                try:
                    with cc1:
                        columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                        columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                        selected_column = st.selectbox("Select a column of interest:", columns_list)
                        time_column = st.selectbox("Select a time column:", columns_list1)
                        col_group = st.multiselect('Group by', current_df.columns, default=[current_df.columns[0], current_df.columns[1]])
                        interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column, col_group)
                        linear_df = interpolation_all.make_interpolation_liner(selected_column, col_group) 
                    
                    with cc2:
                        st.write(" ")
                        st.write(" ")
                        plot_basic = st.button('Plot')    
                        
                    with cc3:
                        st.write(" ")
                        st.write(" ")
                        
                        
                    # Plotting
                    if plot_basic:
                            interpolation_subplot(data_obj.df, linear_df, selected_column, 'linear_fill')
                    
                    # Save results to csv
                    if st.button("Save intermediate linear results"):
                            current_df = linear_df.reset_index()
                            current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)  
                
                # If not time-series compatible dataset
                except KeyError or ValueError or IndexError as e:  
                        st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                        st.stop()
                except  ValueError as ev:  
                        st.warning("Please select different column of interest")
                        st.stop()
                except  IndexError as ei:              
                        st.warning("Please select one more column in group by field")
                        st.stop()
                    
        # Cubic method selected
        if interpolation_radio == 'Cubic':
            # Cubic interpolation method selected
            with st.container():
                st.subheader('Cubic interpolation')
            # Settings layout    
            cc1, cc2, cc3 = st.columns(3)

            # Input settings
            try:
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                    columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                    selected_column = st.selectbox("Select a column of interest:", columns_list)
                    time_column = st.selectbox("Select a time column:", columns_list1)
                    col_group = st.multiselect('Group by', current_df.columns, default=[current_df.columns[0], current_df.columns[1]])
                    interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column, col_group)
                    Cubic_df = interpolation_all.make_interpolation_cubic(selected_column, col_group) 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                
                with cc3:
                     st.write(" ")
                     st.write(" ")
                     
                
                # Plotting
                if plot_basic: 
                   interpolation_subplot(current_df, Cubic_df, selected_column, 'cubic_fill')
                
                # Save results to csv
                if st.button("Save intermediate cubic results"):
                    current_df = Cubic_df.reset_index()
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)   
            
            # If not time-series compatible dataset        
            except KeyError or ValueError or IndexError as e:  
                        st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                        st.stop()
            except  ValueError as ev:  
                        st.warning("Please select different column of interest")
                        st.stop()
            except  IndexError as ei:              
                        st.warning("Please select one more column in group by field")
                        st.stop()
        

        # Forward Fill method selected
        if interpolation_radio == 'Forward Fill':

            # Forward Fill interpolation method selected
            with st.container():
                st.subheader('Forward Fill interpolation')
            
            # Settings layout   
            cc1, cc2, cc3 = st.columns(3)
            
            # Input settings
            try:
                
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                    columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                    selected_column = st.selectbox("Select a column of interest:", columns_list)
                    time_column = st.selectbox("Select a time column:", columns_list1)
                    col_group = st.multiselect('Group by', current_df.columns, default=[current_df.columns[0], current_df.columns[1]])
                    interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column, col_group)
                    df_ffill = interpolation_all.int_df_ffill(selected_column, col_group) 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
              
                with cc3:
                    st.write(" ")
                    st.write(" ")
                
                # Plotting
                if plot_basic:
                   interpolation_subplot(current_df, df_ffill, selected_column, 'Forward Fill')
                
                # Save results to csv
                if st.button("Save intermediate Forward Fill results"):
                    current_df = df_ffill.reset_index()
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False) 
            # If not time-series compatible dataset        
            except KeyError as e: 
                 st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                 st.stop()   
            except  ValueError as ev:  
                        st.warning("Please select different column of interest")
                        st.stop()
            except  IndexError as ei:              
                        st.warning("Please select one more column in group by field")
                        st.stop()

        # Backward Fil method selected
        if interpolation_radio == 'Backward Fill':
            
            # Backward Fil interpolation method selected
            with st.container():
                st.subheader('Backward Fill interpolation')
            
            # Settings layout   
            cc1, cc2, cc3 = st.columns(3)
            
            # Input settings 
            try:
                
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                    columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                    selected_column = st.selectbox("Select a column of interest:", columns_list)
                    time_column = st.selectbox("Select a time column:", columns_list1)
                    col_group = st.multiselect('Group by', current_df.columns, default=[current_df.columns[0], current_df.columns[1]])
                    interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column, col_group)
                    df_bfill = interpolation_all.int_df_bfill(selected_column, col_group) 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    
                with cc3:
                    st.write(" ")
                    st.write(" ")
                  
                
                # Plotting
                if plot_basic: 
                   interpolation_subplot(current_df, df_bfill, selected_column, 'Backward Fill')
                
                # Save results to csv
                if st.button("Save intermediate Backward Fill results"):
                    current_df = df_bfill.reset_index()
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

            # If not time-series compatible dataset                    
            except KeyError as e:
                st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                st.stop()
            except  ValueError as ev:  
                st.warning("Please select different column of interest")
                st.stop()
            except  IndexError as ei:              
                st.warning("Please select one more column in group by field")
                st.stop()       
                
    # Current dataframe display
    with col2:
        st.subheader('Current dataframe')
        st.dataframe(current_df)
        st.write(current_df.shape)

    # Resulting dataframe display
    with col3:
        st.subheader('Resulting dataframe')

        # For each 'Remove outliers' case
        if dp_method == 'Remove outliers' and rmo_radio == 'Std':
            st.dataframe(rm_outlier.reset_index(drop=True))
            st.write(rm_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Q':
            st.dataframe(q_outlier.reset_index(drop=True))
            st.write(q_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Z':
            st.dataframe(z_outlier.reset_index(drop=True))
            st.write(z_outlier.shape)

        # For each 'Smoothing' case
        if dp_method == 'Smoothing' and smooth_radio == 'Median filter':
            st.dataframe(median_filt.reset_index(drop=True))
            st.write(median_filt.shape)
        if dp_method == 'Smoothing' and smooth_radio == 'Moving average':
            st.dataframe(moving_ave.reset_index(drop=True))
            st.write(moving_ave.shape)
        if dp_method == 'Smoothing' and smooth_radio == 'Savitzky Golay':
            st.dataframe(sg.reset_index(drop=True))
            st.write(sg.shape)

        # For each 'Interpolation' case   
        if dp_method == 'Interpolation' and  interpolation_radio == 'Linear':
            st.dataframe(linear_df.reset_index())
            st.write(linear_df.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Cubic':
            st.dataframe(Cubic_df.reset_index())
            st.write(Cubic_df.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Forward Fill':
            st.dataframe(df_ffill.reset_index())
            st.write(df_ffill.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Backward Fill':
            st.dataframe(df_bfill.reset_index())
            st.write(df_bfill.shape)          

    # Changes tracker
    try:
        a = pd.read_csv("Smoothing_and_Filtering//Filtered Dataset.csv")
        if a.equals(current_df):
            st.sidebar.warning("Currently saved results are equal to the current dataframe")
    except:
        st.sidebar.success("You haven't made any changes to the original dataframe yet")
    

    # S&F finalizer
    st.sidebar.subheader("Finalize smoothing & filtering changes:")
    # Create Preprocessing and delete Filtered datasets
    if st.sidebar.button("Finalize!"):
        current_df.to_csv("Smoothing_and_Filtering//Preprocessing dataset.csv", index=False)
        if os.path.isfile("Smoothing_and_Filtering//Filtered Dataset.csv"):
            os.remove("Smoothing_and_Filtering//Filtered Dataset.csv")
        st.sidebar.success("Saved!")

    # Unfinalized changes tracker
    else:
        st.sidebar.error("You have unsaved changes")

# Main
if __name__ == "__main__":
    main()