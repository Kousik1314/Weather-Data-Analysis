import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("weather.csv")
df.dropna(inplace=True)
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Month'] = df['Formatted Date'].dt.month
df['Year'] = df['Formatted Date'].dt.year
df['Temp_Rolling'] = df['Temperature (C)'].rolling(window=30).mean()

monthly_avg = df.groupby('Month')[['Temperature (C)']].mean()

# Start multi-plot figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Weather Data Dashboard', fontsize=16)

# 1. Temperature Trend (Rolling)
axes[0, 0].plot(df['Formatted Date'], df['Temp_Rolling'], color='orange')
axes[0, 0].set_title('30-Day Avg Temperature')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Temp (°C)')
axes[0, 0].grid(True)

# 2. Monthly Avg Temperature
monthly_avg.plot(kind='bar', ax=axes[0, 1], legend=False, color='salmon')
axes[0, 1].set_title('Monthly Avg Temperature')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Temp (°C)')
axes[0, 1].grid(True)

# 3. Precip Type
if 'Precip Type' in df.columns:
    sns.countplot(x='Precip Type', data=df, ax=axes[0, 2], palette='coolwarm')
    axes[0, 2].set_title('Precipitation Type Count')

# 4. Humidity Trend
if 'Humidity' in df.columns:
    axes[1, 0].plot(df['Formatted Date'], df['Humidity'], color='blue', alpha=0.3)
    axes[1, 0].set_title('Humidity Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Humidity')
    axes[1, 0].grid(True)

# 5. Correlation Heatmap
corr = df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 
           'Visibility (km)', 'Pressure (millibars)', 'Temp_Rolling']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 1],
            cbar=True, square=True, annot_kws={"size": 8})

axes[1, 1].set_title('Feature Correlation')

# 6. Hide unused subplot (bottom-right)
axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
