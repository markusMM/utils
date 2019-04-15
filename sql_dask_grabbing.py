import sqlalchemy
import pandas as pd
import tempfile

def make_connectstring(prefix, db, uname, hostname, port):
    """return an sql connectstring"""
    connectstring = prefix + "://" + uname + "@" + hostname + \
                    ":" + port + "/" + db
    return connectstring

def query_to_df(connectstring, query, verbose=False, chunksize=100000):
    """ Return DataFrame from SELECT query and connectstring

    Given a valid SQL SELECT query and a connectstring, return a Pandas 
    DataFrame with the response data.

    Args:
        connectstring: string with connection parameters
        query: Valid SQL, containing a SELECT query
        verbose: prints chunk progress if True. Default False.
        chunksize: Number of lines to read per chunk. Default 100000

    Returns:
        df: A Pandas DataFrame containing the response of query


    """
    
    engine = sqlalchemy.create_engine(
        connectstring, 
        server_side_cursors=True,
        #connect_args=sqlalchemy.make_ssl_args()
        )
    
    # get the data to temp chunk filese
    i = 0
    paths_chunks = []
    with tempfile.TemporaryDirectory() as td:
        for df in pd.read_sql_query(sql=query, con=engine, chunksize=chunksize):
            path = td + "/chunk" + str(i) + ".hdf5"
            df.to_hdf(path, key='data')
            if verbose:
                print("wrote", path)
            paths_chunks.append(path)
            i+=1

        # Merge the chunks using concat, the most efficient way AFAIK
        df = pd.DataFrame()
        for path in paths_chunks:
            df_scratch = pd.read_hdf(path)
            df = pd.concat([df, df_scratch])
            if verbose:
                print("read", path)
    
    return df

###############################################################################
#%%                             -- MAIN --                                  %%#
##%%-----------------------------------------------------------------------%%##
if ( __name__ == '__main__' ):

    connectstring = make_connectstring("postgres")
    
    uname    = "username"
    prefix   = "postgresql"
    db       = "database_name"
    port     = "5432"
    hostname = "hostname_of_db_server"
    connectstring = make_connectstring(prefix, db, uname, hostname, port)

    query = "SELECT * FROM schemaname.tablename"
    df = query_to_df(connectstring, query)