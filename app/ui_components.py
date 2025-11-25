from st_aggrid import AgGrid, GridOptionsBuilder


def aggrid_view(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_grid_options(enableRangeSelection=True, suppressMovableColumns=True)
    gb.configure_default_column(editable=False, filter=True, resizable=True, sortable=True)
    gb.configure_selection("multiple")
    gridOptions = gb.build()

    return AgGrid(
        df,
        gridOptions=gridOptions,
        fit_columns_on_grid_load=True,
        height=450,
        enable_enterprise_modules=True,
        update_mode="MODEL_CHANGED",
        allow_unsafe_jscode=False,
    )
