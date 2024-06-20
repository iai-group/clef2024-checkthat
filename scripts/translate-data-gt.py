import argparse
import csv
from deep_translator import GoogleTranslate
import sys
sys.path.append('.')

def translate_tweets(data_fpath, source_language, target_language):
    """Translate tweets in arabic, dutch or spanish into target language.

    Tweets need to be provided in the format provided in CheckThat Task1

    Args:
        data_fpath: path to input file
        source_language
        target_language
    Output:
        Translated data in the same format as the input format are printed on standar output
    """

    file_df = pd.read_csv(data_fpath, dtype=object, encoding="utf-8", sep='\t')

    text_head = "tweet_text"
    id_head = "tweet_id"
    label = "class_label"
    url = "tweet_url"


    if label in file_df:
        print('{}\t{}\t{}\t{}'.format(id_head,url,text_head,label))
    else:
        print('{}\t{}\t{}'.format(id_head,rl,text_head))

    for i, line in file_df.iterrows():
        text = line["tweet_text"]
        translated_text = GoogleTranslator(
                source=source_language, 
                target=target_language
        ).translate(text=text) 
        
        if label in line:
            print('{}\t{}\t{}\t{}'.format(line[id_head],line[url],translated_text,line[label]))
        else:
            print('{}\t{}\t{}'.format(line[id_head],line[url],translated_text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--file-path", "-f", required=True, type=str, 
            help="The absolute path to the data"
    )
    parser.add_argument("
            --source_language", "-s", required=True, type=str, 
            help="Language of the source"
    )
    parser.add_argument(
            "--target_language", "-t", required=True, type=str, 
            help="Language of the target"
    )

    args = parser.parse_args()
    translate_tweets(args.file_path, args.source_language, args.target_language)

