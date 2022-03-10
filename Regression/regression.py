import datetime
from code import interact
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from .Regression_final import Regressor

#dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



def main(data_obj):
    """_summary_

    Args:
        data_obj (_type_): _description_
    """

    st.header('Regression')

    try:
        var_read = pd.read_csv("Smoothing_and_Filtering//Preprocessing dataset.csv", index_col=None, parse_dates=True, date_parser = pd.to_datetime)
        rg_df = var_read
        for col in rg_df.columns:
            if rg_df[col].dtype == 'object':
                try:
                    rg_df[col] = pd.to_datetime(rg_df[col])
                except ValueError:
                    pass

    except:
        rg_df = data_obj.df.copy()
        for col in rg_df.columns:
            if rg_df[col].dtype == 'object':
                try:
                    rg_df[col] = pd.to_datetime(rg_df[col])
                except ValueError:
                    pass

        st.error("""You did not smooth of filter the data.
                     Please go to 'Smoothing and filtering' and finalize your results.
                     Otherwise, the default dataset would be used!
                     """)   

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    rg_method = st.radio(label='Regression Method', options=['Youssef', 'Richard',
                         'Random Forest'])

    st.dataframe(rg_df)
    st.write(rg_df.shape)
    st.dataframe(rg_df.dtypes.astype(str))
    st.download_button(label="Download data as CSV",
                data=rg_df.to_csv(index=False),
                file_name='Preprocessed Dataset.csv',
                mime='text/csv',
                                            )

    # Selected 'Remove outliers'
    if rg_method == 'Random Forest':

        # st.dataframe(rg_df)
        # st.write(rg_df.shape)

        
        with st.container():
            st.subheader('Select input settings')

            
            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.3, 0.05)
                tree_size = st.slider('Tree size', 100, 1000, 100, 100)
                t_s = st.checkbox("Is this a time-series?")

            with cc2:
                columns_list = list(rg_df.columns)
                if t_s:
                    column_uniques = st.selectbox("Select column with unique values:", columns_list)
                    tm_columns_list = list(rg_df.select_dtypes(include=['datetime']).columns)
                    time_column = st.selectbox("Select a time column:", tm_columns_list)

                selected_column = st.selectbox("Select label column:", columns_list)
                # x_list = columns_list.remove(selected_column)
                x_list = [s for s in columns_list if s != selected_column]


            with cc3:
                if t_s:
                    unique_val = st.selectbox("Select a unique:", list(rg_df[column_uniques].unique()))
                    d = st.date_input("Date:", datetime.date(2022, 2, 24))
                    t = st.time_input('Time:', datetime.time(13, 37))

       
        with st.container():
            X= rg_df[x_list]
            Y= rg_df[selected_column]

            reg_inst = Regressor(X,Y)

            # calling the methods using object 'obj'
            with st.form(key="form"):
                submit_button = st.form_submit_button(label='Submit')

                if submit_button:

                    cc11, cc22, cc33 = st.columns(3)
                    with cc11:
                        with st.spinner("Training models..."):
                            reg_inst.model(tree_size, tt_proportion)

                    reg_inst.result(reg_inst.Y_test, reg_inst.Y_pred)
                    reg_inst.prediction_plot(reg_inst.Y_test, reg_inst.Y_pred)
            

if __name__ == "__main__":
   main()