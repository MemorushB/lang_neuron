import pandas as pd
import os
import matplotlib.pyplot as plt

def make_limited_expert(model_name, language, threshold, base_path='mid_output/Relation/'):

    # file name
    top_file = f'{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{int(threshold/2)}_top.csv'
    bottom_file = f'{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{int(threshold/2)}_bottom.csv'
    both_file = f'{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{threshold}_both.csv'

    if os.path.isfile(top_file) and os.path.isfile(bottom_file) and os.path.isfile(both_file):
        print("expertise_limited files already exist. Skip.")
        return
    
    df = pd.read_csv(f'{base_path}{model_name}/sense/{language}/expertise/expertise.csv')
    print(len(df))

    # Top N
    df2 = df.sort_values('ap', ascending=False)    
    df2 = df2.head(int(threshold/2))
    print(len(df2))
    print(df2.head())

    # Bottom N
    df3 = df.sort_values('ap', ascending=True)    
    df3 = df3.head(int(threshold/2))
    print(len(df3))
    print(df3.head())

    # Top & Bottom
    df4 = pd.concat(
        [df2, df3],
        axis=0,
        ignore_index=True
    )
    print(len(df4))
    print(df4.head())
    
    # Save to files
    df2.to_csv(top_file, index=False)
    df3.to_csv(bottom_file, index=False)
    df4.to_csv(both_file, index=False)
    
    # Plot the distribution and save the plot
    plot_distribution(df2, df3, df4, top_file.replace('.csv', '.png'), bottom_file.replace('.csv', '.png'), both_file.replace('.csv', '.png'))
    
def plot_distribution(df1, df2, df3, topfile, bottomfile, bothfile):

    # Load the new CSVs
    data_top = df1
    data_bottom = df2
    data_both = df3

    # Clean up the column names for each dataset
    data_bottom.columns = data_bottom.columns.str.strip()
    data_top.columns = data_top.columns.str.strip()
    data_both.columns = data_both.columns.str.strip()

    # Function to extract layer number and plot the distribution
    def plot_layer_distribution(data, title, file):
        data['layer_cleaned'] = data['layer'].apply(lambda x: int(x.split('.')[2].replace('layer', '')))
        layer_distribution = data['layer_cleaned'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(layer_distribution.index, layer_distribution.values, color='green')
        plt.xlabel('Layer Number')
        plt.ylabel('Number of Neurons')
        plt.title(f'Distribution of Neurons by Layer: {title}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.savefig(file)

    # Plot distributions for each file
    plot_layer_distribution(data_bottom, 'Bottom 1000 Neurons', bottomfile)
    plot_layer_distribution(data_top, 'Top 1000 Neurons', topfile)
    plot_layer_distribution(data_both, '2000 Neurons Both', bothfile)
    

