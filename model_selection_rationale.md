## Section 3: Related Work and Model Selection

### 3.1 Related Work

Mobile traffic forecasting on the Telecom Italia Milan dataset has been studied extensively since its public release (Barlacchi et al., 2015), spanning classical statistical methods, recurrent deep learning, and more recent convolutional and hybrid architectures.

Classical time-series methods remain a common baseline in this literature. A comparison of deep spatio-temporal neural networks against ARIMA and Holt-Winters exponential smoothing on Milan traffic data found that ARIMA tends to produce almost linear, slowly increasing estimates, while Holt-Winters applies a constant seasonal weight regardless of the prediction step, causing its forecasts to deviate when the data shows frequent fluctuation. This is a well-documented limitation of linear seasonal models when traffic exhibits bursts or irregular deviations from its typical seasonal template.

Recurrent neural networks are the most widely applied deep learning approach for this dataset specifically. A study comparing LSTM and GRU architectures, alongside Random Forest and Decision Tree baselines, on two months of Telecom Italia Milan data found that both deep learning models were effective at capturing Internet activity and seasonality, with the deep learning models outperforming the conventional machine learning baselines and LSTM outperforming GRU overall. Performance was also found to vary across geographic clusters within the city, suggesting that a single architecture may not generalize uniformly across areas with different traffic characteristics — directly relevant to this assignment's research question.

More recent work has explored convolutional and hybrid temporal architectures as alternatives to purely recurrent models. A hybrid multi-TCN-LSTM model applied to mobile traffic data was shown to capture spatiotemporal characteristics more accurately than conventional MLP and LSTM models, as confirmed by MAPE and MAE results. Temporal Convolutional Networks process an input sequence via dilated causal convolutions rather than sequential recurrence, allowing full parallelization during training and a receptive field that can be tuned directly to match known periodicities in the data (e.g., a 24-hour or 7-day span), rather than relying on a recurrent cell to propagate that information step by step.

### 3.2 Selected Models

Based on this literature and the characteristics identified in Section 2 (strong daily and weekly autocorrelation, a stationary STL residual, heterogeneous traffic scale and regularity across areas, and localized anomalies such as the traffic spike observed in Square 5161), three models were selected for implementation:

**Seasonal ARIMA (SARIMA).** [Cite your own ACF result here — e.g., "ACF analysis in Section 2 showed autocorrelation of X at the 144-step (daily) lag and Y at the 1008-step (weekly) lag, with the STL residual passing the ADF test (p = Z), indicating the deseasonalized series is close to stationary."] These conditions are close to the assumptions SARIMA is built on, making it a principled classical baseline rather than an arbitrary one. Its expected limitation, informed by the literature above, is an inability to react to the kind of localized, non-repeating spike seen in Square 5161 — its seasonal component is fixed and does not adapt within a forecast window.

**Long Short-Term Memory (LSTM).** Chosen because it is the strongest-performing recurrent architecture reported specifically on this dataset in prior work, and because it can model nonlinear temporal dependencies that a linear model cannot — relevant given the traffic bursts and the weekday/weekend irregularity observed in Square 5259's lower weekly-seasonality-strength score relative to the other top-traffic areas. Its known limitation is a dependence on having sufficient training data per area; this is worth monitoring given the traffic-volume heterogeneity found in the distribution analysis, where most of the 10,000 areas carry very little total traffic.

**Temporal Convolutional Network (TCN).** Selected as an architecturally distinct third model — convolutional and parallelizable rather than recurrent — motivated by evidence that TCN-based hybrids outperform plain LSTM on mobile traffic data of this kind. Beyond potential accuracy gains, a TCN is expected to train faster than an LSTM of comparable capacity, which is directly relevant to the training/execution time comparison required in Section 4.

### 3.3 Summary

These three models span linear statistical, recurrent nonlinear, and convolutional nonlinear paradigms, satisfying the requirement for architecturally distinct comparison points. Each choice is tied to a specific empirical finding from Section 2, and each carries a documented limitation from the literature that motivates part of the comparative discussion required in Section 4 (e.g., testing whether SARIMA underperforms specifically around the Square 5161 anomaly, or whether LSTM's advantage over SARIMA narrows on lower-traffic, sparser areas).

### References

- Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G., Antonelli, F., Vespignani, A., Pentland, A., & Lepri, B. (2015). A multi-source dataset of urban life in the city of Milan and the Province of Trentino. _Scientific Data_, 2, 150055.
- Santos, G. L., Rosati, P., Lynn, T., Kelner, J., Sadok, D., & Endo, P. T. (2022). Predicting short-term mobile Internet traffic from Internet activity using recurrent neural networks. _International Journal of Network Management_, 32(3), e2191.
- Zhang, C., & Patras, P. (2018). Long-term mobile traffic forecasting using deep spatio-temporal neural networks. In _Proceedings of the Eighteenth ACM International Symposium on Mobile Ad Hoc Networking and Computing_ (pp. 231-240).
- Park, J., Mwasinga, L. J., Yang, H., Raza, S. M., Le, D.-T., Kim, M., Chung, M. Y., & Choo, H. (2024). Regional correlation aided mobile traffic prediction with spatiotemporal deep learning. In _2024 IEEE Consumer Communications & Networking Conference (CCNC)_ (pp. 566-569).

---

A few things to do before this goes in your report:

- Fill in the bracketed placeholders with your actual ACF/ADF numbers from the seasonality script output.
