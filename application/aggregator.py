"""
Aggregates the data from all databases into one CSV with minimal cutting/processing
"""
import sqlite3
import csv
import pandas as pd

HEADERS = list(['track_id', 'title', 'song_id', 'release', 'artist_id', 'artist_mbid',
                'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss',
                'year', 'genre', 'lyrics'])


def main():
    """
    Executes the aggregation procedure pulling data from databases into one CSV.
    :return:
    """
    pd.DataFrame(columns=HEADERS).to_csv('../aggregated.csv')

    with open('../tagtraum_genre.cls', mode='r') as genre_map:
        genre_reader = csv.DictReader(genre_map, delimiter='\t')
        with sqlite3.connect('../musixmatch_lyrics.db') as lyrics_db:
            lyrics_c = lyrics_db.cursor()
            words_c = lyrics_db.cursor()
            with sqlite3.connect('../msd_meta.db') as msd_db:
                msd_c = msd_db.cursor()

                # Restriction for testing purposes
                total_count = 0
                valid_count = 0

                for row in genre_reader:
                    total_count += 1
                    print()
                    print("Entry # {}".format(total_count))
                    msd_c.execute('SELECT * FROM songs WHERE track_id LIKE ?', [row['msd_id']])
                    song = msd_c.fetchall()
                    if not song:
                        continue

                    print(row['msd_id'])

                    # Make a list out of the song metadata
                    params_towrite = list(song[0])
                    params_towrite.append(row['genre'])
                    print(params_towrite[1])

                    # Get the lyrics
                    lyrics_towrite = ''
                    lyrics_c.execute('SELECT word, count FROM lyrics WHERE track_id LIKE ?',
                                     [row['msd_id']])
                    lyrics = lyrics_c.fetchall()

                    # Skip if we don't have lyrics
                    if not lyrics:
                        print("No lyrics, skipping")
                        continue

                    # Convert the lyrics into the desired format
                    for lyric in lyrics:
                        words_c.execute('SELECT number FROM words_indexed WHERE word LIKE ?',
                                        [lyric[0]])
                        lyrics_towrite += '{}:{} '.format(words_c.fetchone()[0], lyric[1])

                    # Add the lyrics as the last parameter
                    params_towrite.append(lyrics_towrite)

                    # Restrict for testing purposes
                    valid_count += 1
                    print("Valid entry # {}".format(valid_count))
                    with open('../aggregated.csv', mode='a') as out_csv:
                        # Output to CSV
                        pd.DataFrame(data=[params_towrite], columns=HEADERS).to_csv(out_csv, header=False)

                    if valid_count >= 10:
                        break


if __name__ == "__main__":
    main()
