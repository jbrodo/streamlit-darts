import pandas as pd
from darts import TimeSeries
import streamlit as st

def main():
    # Read a pandas DataFrame
    df = pd.read_csv("streamlit_darts/AirPassengers.csv", delimiter=",")

    # Create a TimeSeries, specifying the time and value columns
    series = TimeSeries.from_dataframe(df, "Month", "#Passengers")

    # Set aside the last 36 months as a validation series
    train, val = series[:-36], series[-36:]

    st.title("Exponential Smooting")
    from darts.models import ExponentialSmoothing

    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=1000)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()

    st.pyplot(fig)

    st.title("ARIMA")
    from darts.models import ARIMA

    model = ARIMA()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=1000)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()

    st.pyplot(fig)

    st.title("Auto ARIMA")
    from darts.models import AutoARIMA

    model = AutoARIMA()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=1)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()

    st.pyplot(fig)

    from darts.datasets import ETTh2Dataset

    series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]
    train, val = series.split_before(0.6)

    from darts.ad import KMeansScorer

    scorer = KMeansScorer(k=2, window=5)
    scorer.fit(train)
    anom_score = scorer.score(val)

    st.title("Quantile Detector using ETTH2Dataset and KMeansScorer")
    from darts.ad import QuantileDetector

    detector = QuantileDetector(high_quantile=0.99)
    detector.fit(scorer.score(train))
    binary_anom = detector.detect(anom_score)


    fig = plt.figure()

    series.plot()
    (anom_score / 2. - 100).plot(label="computed anomaly score", c="orangered", lw=3)
    (binary_anom * 45 - 150).plot(label="detected binary anomaly", lw=4)

    st.pyplot(fig)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Dart Examples", page_icon=":chart_with_upwards_trend:"
    )
    
    main()