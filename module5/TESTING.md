# Dashboard verification

Verified October 17, 2026 using Python 3.12, scikit-learn 1.8.0, Streamlit 1.50.0 and Fairlearn 0.13.0.

- Six automated tests passed.
- The 2,800 archived final-test rows reproduce the Module 4 model and baseline MAE, RMSE, WAPE and aggregate bias.
- Fairlearn group losses agree with the original store/department review.
- All three prediction examples reproduce their saved validation predictions and SHAP sums.
- The original Module 4 API sample reproduces 0.8481716946876521 daily sales units.
- Invalid product/store combinations and negative history inputs return HTTP 422.
- Streamlit AppTest exercised all five views, store filtering, an empty selection, example switching, what-if changes and department comparisons.
- A separate live local HTTP check returned 200 for GET /health and POST /predict. See HTTP_API_VERIFICATION.json.

The tests did not retrain the model or create new final-test forecasts. Displayed errors are recomputed from archived predictions for filtering and comparison.

Public hosting has not been deployed or verified. The final dashboard URL must be tested independently after deployment.
