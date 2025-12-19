import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import asyncio
import httpx
import time
from concurrent.futures import ProcessPoolExecutor

'''
1. сетевые запросы: синхронный и асинхронный
асинхронный способ оказался чуть медленнее
чтобы его сделать один асинхронный запрос нужно проделать несколько действий создать цикл, подготовить корутину, это занимает дополнительное время

2. обработка данных: последовательная и параллельная
здесь похожая ситуация: нужно запустить новые копии питона на других ядрах процессора, скопировать данные в эти процессы, после расчётов собрать результаты обратно

из-за того что тут маловато данных подготовка всего занимает дольше чем профит от этих методов 

'''

st.set_page_config(page_title="Temperature Monitoring System", layout="wide")

# анализ
def analyze_city(city_df):
    city_df = city_df.sort_values('timestamp')
    city_df['rolling_mean'] = city_df['temperature'].rolling(window=30, center=True).mean()
    
    seasonal_stats = city_df.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    seasonal_stats.columns = ['season', 'mean_season', 'std_season']
    
    city_df = city_df.merge(seasonal_stats, on='season', how='left')
    city_df['is_anomaly'] = (city_df['temperature'] < (city_df['mean_season'] - 2 * city_df['std_season'])) | \
                            (city_df['temperature'] > (city_df['mean_season'] + 2 * city_df['std_season']))
    return city_df

def run_sequential(df):
    start = time.time()
    results = [analyze_city(df[df['city'] == city]) for city in df['city'].unique()]
    return pd.concat(results), time.time() - start

def run_parallel(df):
    start = time.time()
    city_groups = [df[df['city'] == city] for city in df['city'].unique()]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(analyze_city, city_groups))
    return pd.concat(results), time.time() - start

# апи
def get_weather_sync(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    start = time.time()
    try:
        response = requests.get(url, timeout=5)
        return response.status_code, response.json(), time.time() - start
    except Exception as e:
        return 500, {"message": str(e)}, time.time() - start

async def get_weather_async(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    start = time.time()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5)
            return response.status_code, response.json(), time.time() - start
        except Exception as e:
            return 500, {"message": str(e)}, time.time() - start

# данные
st.title("Аналитическая система мониторинга температур")

col_setup1, col_setup2 = st.columns(2)
with col_setup1:
    uploaded_file = st.file_uploader("Загрузите файл с историческими данными (CSV)", type="csv")
with col_setup2:
    api_key = st.text_input("Введите API ключ OpenWeatherMap", type="password")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    processed_data, _ = run_parallel(data)

    # города
    st.divider()
    cities = sorted(data['city'].unique())
    selected_city = st.selectbox("Выберите город для анализа", cities)
    
    city_data = processed_data[processed_data['city'] == selected_city]

    # статистика
    st.header(f"Историческая статистика по сезонам: {selected_city}")
    
    season_order = ['winter', 'spring', 'summer', 'autumn']
    stats = city_data.groupby('season')['temperature'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats['season'] = pd.Categorical(stats['season'], categories=season_order, ordered=True)
    stats = stats.sort_values('season').set_index('season')

    st.dataframe(
        stats,
        use_container_width=True,
        column_config={
            "count": st.column_config.NumberColumn("Дней", format="%d"),
            "mean": st.column_config.NumberColumn("Средняя (°C)", format="%.2f"),
            "std": st.column_config.NumberColumn("Отклонение", format="%.2f"),
            "min": st.column_config.NumberColumn("Мин (°C)", format="%.2f"),
            "max": st.column_config.NumberColumn("Макс (°C)", format="%.2f")
        }
    )

    # тек. температура
    st.divider()
    st.header("Мониторинг текущей температуры")
    
    if api_key:
        status, weather_res, _ = get_weather_sync(selected_city, api_key)
        
        if status == 200:
            current_temp = weather_res['main']['temp']
            curr_month = pd.Timestamp.now().month
            month_to_season = {12: "winter", 1: "winter", 2: "winter",
                               3: "spring", 4: "spring", 5: "spring",
                               6: "summer", 7: "summer", 8: "summer",
                               9: "autumn", 10: "autumn", 11: "autumn"}
            curr_season = month_to_season[curr_month]
            
            if curr_season in stats.index:
                h_mean = stats.loc[curr_season, 'mean']
                h_std = stats.loc[curr_season, 'std']
                
                col_met1, col_met2 = st.columns(2)
                col_met1.metric("Текущая температура", f"{current_temp} °C")
                col_met2.write(f"Норма для сезона {curr_season}: {h_mean:.2f} °C (± {2*h_std:.2f} °C)")
                
                if (current_temp < h_mean - 2*h_std) or (current_temp > h_mean + 2*h_std):
                    st.error(f"Внимание: Температура {current_temp} °C аномальна для сезона {curr_season}.")
                else:
                    st.success(f"Температура {current_temp} °C в пределах сезонной нормы.")
        else:
            st.error("Ошибка API при получении текущей погоды.")

    # временной ряд
    st.divider()
    st.header("Динамика изменений и аномалии")
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['temperature'],
                                mode='lines', name='Ежедневно', 
                                line=dict(color='LightSteelBlue', width=1)))
    fig_ts.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['rolling_mean'],
                                mode='lines', name='Тренд (30 дней)', 
                                line=dict(color='SlateGray', width=2)))
    
    anomalies = city_data[city_data['is_anomaly']]
    fig_ts.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['temperature'],
                                mode='markers', name='Аномалия', 
                                marker=dict(color='IndianRed', size=5)))
    
    fig_ts.update_layout(plot_bgcolor='white', xaxis_title="Дата", yaxis_title="°C")
    st.plotly_chart(fig_ts, use_container_width=True)

    # сезоны
    st.header("Сезонные профили")
    
    seasonal_plot_data = stats.reset_index()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=seasonal_plot_data['season'],
        y=seasonal_plot_data['mean'],
        error_y=dict(type='data', array=seasonal_plot_data['std'], visible=True),
        marker_color='LightSteelBlue',
        name='Среднее'
    ))
    
    fig_bar.update_layout(
        plot_bgcolor='white',
        xaxis_title="Сезон",
        yaxis_title="Средняя температура (°C)",
        xaxis={'categoryorder':'array', 'categoryarray': season_order}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # скорость обработки
    st.divider()
    st.header("Технические показатели производительности")
    
    # запросы сети
    if api_key:
        _, _, t_sync = get_weather_sync(selected_city, api_key)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _, _, t_async = loop.run_until_complete(get_weather_async(selected_city, api_key))
        except:
            t_async = t_sync
            
        st.subheader("Сетевые запросы к API (запрос одного города)")
        cn1, cn2 = st.columns(2)
        cn1.metric("Синхронный метод", f"{t_sync:.4f} s")
        
        # разница
        if t_async < t_sync:
            n_diff = ((t_sync - t_async) / t_sync) * 100
            cn2.metric("Асинхронный метод", f"{t_async:.4f} s", delta=f"{n_diff:.1f}% быстрее")
        else:
            n_diff = ((t_async - t_sync) / t_sync) * 100
            cn2.metric("Асинхронный метод", f"{t_async:.4f} s", delta=f"{n_diff:.1f}% медленнее", delta_color="inverse")
    
    # процессорные
    _, t_seq = run_sequential(data)
    _, t_par = run_parallel(data)
    
    st.subheader("Обработка данных процессором (весь массив данных)")
    cc1, cc2 = st.columns(2)
    cc1.metric("Последовательная обработка", f"{t_seq:.3f} s")
    
    if t_par < t_seq:
        p_diff = ((t_seq - t_par) / t_seq) * 100
        cc2.metric("Параллельная обработка", f"{t_par:.3f} s", delta=f"{p_diff:.1f}% быстрее")
    else:
        p_diff = ((t_par - t_seq) / t_seq) * 100
        cc2.metric("Параллельная обработка", f"{t_par:.3f} s", delta=f"{p_diff:.1f}% медленнее", delta_color="inverse")

else:
    st.info("Пожалуйста, загрузите CSV файл.")
