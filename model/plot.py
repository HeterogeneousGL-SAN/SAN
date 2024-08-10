import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

def plot():
    # Step 1: Prepare the data
    data = {
        'Method': ['GraphSAGE', 'GAT', 'HAN', 'HGT', 'VGAE', 'Metapath2vec', 'SchAttNet'],
        'AUC_f': [0.9495, 0.9662, 0.8856, 0.9502, 0.9515, 0.5601, 0.9710]
    }

    # Step 2: Create a DataFrame
    df = pd.DataFrame(data)

    # Step 3: Sort the DataFrame by AUC values (descending order)
    #df = df.sort_values(by='AUC_Full_Metadata', ascending=False)

    # Step 4: Plot the bar chart using Seaborn with adjusted dodge parameter
    plt.figure(figsize=(4, 5))
    sns.barplot(x='Method', y='AUC_f', data=df, palette='Set2', dodge=0.7)  # Adjust dodge as needed
    patterns = ['...', '///', 'xxx', '\\\\']
    bars = ax.patches

    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)
        bar.set_edgecolor('lightgrey')  # Cambiamo il colore del bordo a grigio chiaro
        bar.set_facecolor('white')
    #plt.title('AUC Values by Method (Full Metadata)')
    #plt.xlabel('Method')
    #plt.ylabel('AUC')
    #plt.ylim(0.5, 1.0)  # Set the limits for y-axis for better visualization
    #plt.xticks(rotation=45)  # Rotate the method names for better readability
    plt.show()


import os
import shutil


def move_files_with_4096(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    print('dentro')
    # Iterate over all the files in the source folder
    files = [file for file in os.listdir(source_folder) if file.endswith('pt')]
    print(len(files))
    for filename in files:
        if 'NO_METADATA'  in filename and '4096' in filename and 'heads-8' and 'cores-5' and 'concat' in filename and 'full' not in filename:
            # Construct full file paths
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)

            # Move the file
            shutil.move(source_file, destination_file)
            print(f"Moved: {source_file} to {destination_file}")


    # Example usage
    source_folder = './model/checkpoint/transductive/models/pubmed/enriched_all/'
    destination_folder = './model/checkpoint/transductive/models/pubmed/enriched_all/san/no_metadata/'
    move_files_with_4096(source_folder, destination_folder)

    source_folder = './model/checkpoint/inductive/models/mes/bootstrapped/50/'
    destination_folder = './model/checkpoint/inductive/models/pubmed/enriched_all/san/full/metadata/'
    #move_files_with_4096(source_folder, destination_folder)

    source_folder = './model/checkpoint/inductive/models/mes/bootstrapped/75/'
    destination_folder = './model/checkpoint/inductive/models/mes/bootstrapped/san/full/75/'
    #move_files_with_4096(source_folder, destination_folder)


import pandas as pd
import os
def print_res():


    # Specifica il percorso della cartella contenente i file CSV
    cartella = f'./datasets/pubmed/split_transductive/train/'

    # Elenco tutti i file nella cartella
    file_list = os.listdir(cartella)

    # Filtra solo i file CSV
    csv_files = [f for f in file_list if f.endswith('.csv')]

    # Itera attraverso ogni file CSV e stampa il numero di righe
    for file in csv_files:
        percorso_file = os.path.join(cartella, file)
        try:
            # Legge il file CSV
            df = pd.read_csv(percorso_file)
            # Stampa il numero di righe
            numero_righe = len(df)
            print(f'File: {file} - Numero di righe: {numero_righe}')
        except Exception as e:
            print(f'Errore durante la lettura del file {file}: {e}')

print_res()
