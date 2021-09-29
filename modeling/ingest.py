from urllib.request import urlretrieve
import os
import sqlite3

class SpotifyDatabaseIngestor:
    """ A class to ingest the external raw SQL instructions into a SQLite db.

    """

    def __init__(self, source_url:str, db_path:str, db_name:str):
        """ Create a SpotifyDatabaseIngestor instance.

        Args:
            source_url (str): path where the SQL file can be downloaded.
            db_path (str): path to the local directory where the db should live.
            db_name (str): file name for the local db.

        """
        self.source_url = source_url
        self.db_path = db_path
        self.db_name = db_name
    
    def retrieve_db_file(self, file_name:str="temp.sql", replace:bool=True) -> None:
        """ Downloads the database file to file_name.

        Args:
            file_name (str): path to place raw SQL file.
            replace (bool): should the raw file be replaced if it exists?

        """
        if not replace:
            if os.path.exists(self.db_path+file_name):
                raise FileExistsError()
        if not os.path.isdir(self.db_path):
            os.mkdir(self.db_path)
        urlretrieve(self.source_url, self.db_path+file_name)

    def create_database_from_file(self, file_name:str="temp.sql", replace:bool=True, remove_sql:bool=True) -> None:
        """ Builds a database using the downloaded SQL file.

        Args:
            file_name (str): path of raw SQL file.
            replace (bool): should the database be replaced if it exists?
            remove_sql (bool): should the SQL file be deleted after the db is built?

        """
        con = sqlite3.connect(self.db_path+self.db_name) 
        cur = con.cursor()
        if not replace:
            if os.path.exists(self.db_path+self.db_name):
                raise FileExistsError()
        if not os.path.exists(self.db_path+file_name):
            raise FileNotFoundError()
        clause = ""
        with open(self.db_path+file_name,"r", encoding='utf8') as sql_file:
            for line in sql_file:
                # SQLite doesn't use the KEY lines MySQL does
                if "  KEY" not in line and "UNIQUE KEY `unique_index`" not in line:
                    # SQLite wants the FOREIGN KEY lines translated
                    if "CONSTRAINT" in line:
                        line = line[line.find("FOREIGN"):].replace(" ON DELETE NO ACTION ON UPDATE NO ACTION","").replace("FOREIGN KEY ","FOREIGN KEY")
                    clause += line
                    # Execute lines with a ;
                    if ";" in clause:
                        # Don't execute LOCK TABLES clauses
                        if "LOCK TABLES" not in clause:
                            # Get rid of these chars in every line
                            clause = clause.replace("ENGINE=InnoDB DEFAULT CHARSET=latin1","").replace("\\\\","").replace("\\'","").replace('\\"',"").replace(",\n)",")")
                            cur.execute(clause)
                            con.commit()
                        clause = ""
        if remove_sql:
            os.remove(self.db_path+file_name)