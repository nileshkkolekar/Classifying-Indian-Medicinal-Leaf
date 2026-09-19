"""Streamlit front end for the prediction API.

Run with::

    streamlit run src/medicinal_leaf/ui/streamlit_app.py

The UI is a pure client: it talks to the FastAPI service over HTTP and holds
no model of its own, so the two can be deployed and scaled separately.
"""
