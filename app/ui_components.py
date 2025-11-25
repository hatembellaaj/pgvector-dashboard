from st_aggrid import AgGrid, GridOptionsBuilder


def aggrid_view(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(editable=False, filter=True)
    gb.configure_selection("single")
    gridOptions = gb.build()

    return AgGrid(
        df,
        gridOptions=gridOptions,
        fit_columns_on_grid_load=True,
        height=400,
    )
