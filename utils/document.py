import glob
import os
import shutil

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


def delete_uploaded_files_and_db():
    '''
    deletes all files stored in files directory
    and deletes db directory
    '''

    # get uploaded files path
    file_paths = glob.glob('files/*')

    if file_paths:
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete file: {e}")
    shutil.rmtree("db")