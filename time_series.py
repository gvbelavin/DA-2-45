import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def generate_time_series(start_date='2010-01-01', num_periods=1000, seasonality_period=365):
    
    #Эта функция генерирует синтетический временной ряд с сезонностью, а возвращает DataFrame или None при ошибке
    
    try:
        # Проверка входных параметров
        if not isinstance(start_date, str):
            raise ValueError("start_date должен быть строкой в формате даты (YYYY-MM-DD)")
        if not isinstance(num_periods, int) or num_periods <= 0:
            raise ValueError("num_periods должен быть положительным целым числом")
        if not isinstance(seasonality_period, int) or seasonality_period <= 0:
            raise ValueError("seasonality_period должен быть положительным целым числом")
        
        pd.to_datetime(start_date)
        np.random.seed(42)
        
        # Генерирум дату
        dates = pd.date_range(start=start_date, periods=num_periods, freq='D')
        
        # Генерация временного ряда
        time = np.arange(num_periods)
        seasonal_component = np.sin(2 * np.pi * time / seasonality_period)
        noise = np.random.normal(0, 0.5, num_periods)
        time_series = seasonal_component + noise
        
        return pd.DataFrame({'value': time_series}, index=dates)
    
    except Exception as e:
        print(f"Ошибка при генерации данных: {str(e)}")
        return None

def plot_time_series(df):
    
    #Визуализирует исходный временной ряд
    try:
        if df is None:
            raise ValueError("DataFrame не был создан")
        if df.empty:
            raise ValueError("DataFrame пустой")
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['value'])
        plt.title('Синтетический временной ряд с сезонностью')
        plt.xlabel('Дата')
        plt.ylabel('Значение')
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Ошибка при визуализации: {str(e)}")

def decompose_and_plot(df, seasonality_period):
    
    #Разделяет ряд на компоненты и визуализирует результаты
    try:
        if df is None:
            raise ValueError("DataFrame не был создан")
        if df.empty:
            raise ValueError("DataFrame пустой")
        if len(df) < seasonality_period:
            raise ValueError(f"Период сезонности ({seasonality_period}) превышает длину ряда ({len(df)})")
        
        # Проверка на стабильность периода
        if seasonality_period <= 0:
            raise ValueError("Период сезонности должен быть положительным")
        
        decomposition = seasonal_decompose(
            df['value'], 
            model='additive', 
            period=seasonality_period
        )
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        components = [
            decomposition.observed, 
            decomposition.trend, 
            decomposition.seasonal, 
            decomposition.resid
        ]
        titles = ['Наблюдаемые значения', 'Тренд', 'Сезонность', 'Остатки']
        
        for ax, component, title in zip(axes, components, titles):
            component.plot(ax=ax)
            ax.set_title(title)
            ax.set_xlabel('')
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Ошибка при декомпозиции: {str(e)}")

def main():
    df = generate_time_series()
    plot_time_series(df)
    decompose_and_plot(df, seasonality_period=365)

if __name__ == "__main__":
    main()
