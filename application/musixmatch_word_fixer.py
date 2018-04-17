"""
Adds a "row number" column into the "words" table in the musiXmatch database
"""
import sqlite3


def main():
    """
    Adds a "row number" column into the "words" table in the musiXmatch database
    :return:
    """
    with sqlite3.connect('../musixmatch_lyrics.db') as lyrics_db:
        lyrics_c = lyrics_db.cursor()
        lyrics_writec = lyrics_db.cursor()
        # Remove the existing table, if it exists
        try:
            lyrics_c.execute('DROP TABLE words_indexed')
        except sqlite3.OperationalError:
            pass

        # Create the new table with the added column
        lyrics_c.execute('CREATE TABLE words_indexed(number INTEGER,word TEXT)')

        # Get all words from original table and add the index column
        lyrics_c.execute('SELECT * FROM words')
        count = 1
        for word in lyrics_c.fetchall():
            lyrics_writec.execute('INSERT INTO words_indexed VALUES(?, ?)', (count, word[0]))
            count += 1

        lyrics_db.commit()


if __name__ == "__main__":
    main()
