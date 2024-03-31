import datetime
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.font_manager import fontManager

from donut_snake.LoadAndTest_class import DonutSnake

st.markdown("# :sunglasses: 跨平台股票歷史價格預測程式")

st.link_button("網站", "https://github.com/ccpl17/donut-snake")

a = None


def hello():
    global a

    _ticker = st.session_state.ticker

    match st.session_state.locale:
        case "台灣":
            _ticker += ".TW"
        case "香港":
            _ticker += ".HK"

    a = DonutSnake(ticker=_ticker,
                   # 期間要大於回看天數
                   start_date=st.session_state.start_date,
                   end_date=st.session_state.end_date,
                   look_back=30,
                   epochs=st.session_state.epochs,
                   batch_size=st.session_state.batch_size,
                   back_month=st.session_state.back_month)

    # 添加下載按鈕
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('UTF-8')

    csv = convert_df(a.output)

    st.download_button(
        "下載結果",
        csv,
        f"{_ticker}" + f"_{start_date}" + f"_{end_date}" + ".csv",
        "text/csv",
        key='download-csv'
    )

    st.markdown(f"""\
### 模型評估

- MSE : {a.mse}

- RMSE : {a.rmse}

- MAE :{a.mae}

- R-Squared : {a.r_squared}

""")

    fontManager.addfont("jf-openhuninn-2.0.ttf")
    matplotlib.rc("font", family="jf-openhuninn-2.0")

    plt.figure(figsize=(10, 6))
    plt.plot(a.output['real stock price'], label='真實股價')
    plt.plot(a.output['Predict stock price'], label='預測股價')
    plt.title(f"{a.ticker}股價預測")
    plt.xlabel("時間")
    plt.ylabel("股價")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


with st.form("my_form"):
    # 股票代碼
    if "ticker" not in st.session_state:
        st.session_state.ticker = None
    st.text_input("🫵輸入股票代碼", key="ticker")

    # 選地區
    st.selectbox(
        '🫵選擇股票地區',
        ('台灣', '美國', '香港'),
        key="locale",
    )

    # 取得當前日期
    current_date = datetime.date.today()

    # 計算一個月前的日期
    one_month_ago = current_date - datetime.timedelta(days=60)

    # Streamlit 中的日期選擇器
    start_date = st.date_input("🫵選擇開始日期", value=one_month_ago, key="start_date")
    end_date = st.date_input("🫵選擇結束日期", value=current_date, key="end_date")

    if start_date > end_date:
        st.markdown('<span style="color:red">開始日期不能晚於結束日期，請重新選擇。</span>', unsafe_allow_html=True)

    # 添加提交按鈕
    st.form_submit_button("提交", on_click=hello)

    with st.expander("進階選項"):

        st.slider('epochs', key="epochs", min_value=10, max_value=300, value=100, step=10)
        st.slider('batch_size', key="batch_size", min_value=32, max_value=1024, value=32, step=32)
        st.slider('back_month', key="back_month", min_value=1, max_value=36, value=12, step=1)
        st.slider('look_back', key="look_back", min_value=5, max_value=60, value=30, step=1)

        st.image("https://th.bing.com/th/id/OIP.NnaPW7OADa9O6vgXqVN8PgHaHM?rs=1&pid=ImgDetMain")
        st.image(
            "https://th.bing.com/th/id/R.e7562747d7bf50e53109468f2f8f8f01?rik=nznA0ML9keVZuw&riu=http%3a%2f%2fmemenow.cc%2fwp-content%2fuploads%2f2020%2f04%2f20200408_5e8e5149f3152.jpg&ehk=POD5ktlUpU%2fnBk%2flkyF1w7QW6y%2fNVhfkZsg66e6oDYw%3d&risl=&pid=ImgRaw&r=0")
        st.image("https://pic4.zhimg.com/50/v2-9cc449d9e18d63df5377401f8a508a01_hd.jpg")
