"""
Adds a "row number" column into the "words" table in the musiXmatch database
"""
import sqlite3

with sqlite3.connect('../musixmatch_lyrics.db') as lyrics_db:
    LYRICS_C = lyrics_db.cursor()
    LYRICS_WRITEC = lyrics_db.cursor()
    try:
        LYRICS_C.execute('DROP TABLE words_indexed')
    except sqlite3.OperationalError:
        pass

    LYRICS_C.execute('CREATE TABLE words_indexed(number INTEGER,word TEXT)')

    LYRICS_C.execute('SELECT * FROM words')
    COUNT = 1
    for word in LYRICS_C.fetchall():
        LYRICS_WRITEC.execute('INSERT INTO words_indexed VALUES(?, ?)', (id, word[0]))
        COUNT += 1

    lyrics_db.commit()
