import glob
import os

def process_file_multi_docs(files):
        '''
        gets all files and writes bytes data in a directory
        returns list of saved files
        '''

        # Create file directory
        if not os.path.exists('files'):
            os.makedirs('files')

        for file in files:
            bytes_data = file.read()

            # _, file_extension = os.path.splitext(file.name)
            # st.write(_, file_extension)

            # Write uploaded file to disk
            with open(f'files/{file.name}', 'wb') as f:
                f.write(bytes_data)

        # get uploaded files path
        file_paths = glob.glob('files/*')

        return file_paths