"""
Aggregates the data from all databases into one CSV with minimal cutting/processing
"""
import sqlite3
import csv
import pandas as pd

HEADERS = list(['track_id', 'title', 'song_id', 'release', 'artist_id', 'artist_mbid',
                'artist_name', 'duration', 'artist_familiarity', 'artist_hotttnesss',
                'year', 'lyrics'])

DATA = []

with open('../tagtraum_genre.cls', mode='r') as genre_map:
    GENRE_READER = csv.DictReader(genre_map, delimiter='\t')
    with sqlite3.connect('../musixmatch_lyrics.db') as lyrics_db:
        LYRICS_C = lyrics_db.cursor()
        WORDS_C = lyrics_db.cursor()
        with sqlite3.connect('../msd_meta.db') as msd_db:
            MSD_C = msd_db.cursor()

            # Restriction for testing purposes
            COUNT = 0

            for row in GENRE_READER:
                print(row['msd_id'])
                MSD_C.execute('SELECT * FROM songs WHERE track_id LIKE ?', [row['msd_id']])
                song = MSD_C.fetchall()
                if not song:
                    continue

                # Make a list out of the song metadata
                params_towrite = list(song[0])
                print(params_towrite[1])

                # Get the lyrics
                lyrics_towrite = ''
                LYRICS_C.execute('SELECT word, count FROM lyrics WHERE track_id LIKE ?',
                                 [row['msd_id']])
                lyrics = LYRICS_C.fetchall()

                # Convert the lyrics into the desired format
                for lyric in lyrics:
                    WORDS_C.execute('SELECT number FROM words_indexed WHERE word LIKE ?',
                                    [lyric[0]])
                    lyrics_towrite += '{}:{} '.format(WORDS_C.fetchone()[0], lyric[1])

                # Add the lyrics as the last parameter
                params_towrite.append(lyrics_towrite)
                DATA.append(params_towrite)

                # Restrict for testing purposes
                COUNT += 1
                if COUNT >= 10:
                    break

    # Output to CSV
    pd.DataFrame(data=DATA, columns=HEADERS).to_csv('../aggregated.csv')
