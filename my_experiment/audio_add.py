# to run thi code, please download ffmmpeg and configure environment variable
import subprocess
import os
from pydub import AudioSegment 



def video_add_audio(video_file, audio_file, outfile_name='current_add_audio_2.mp4'):
    """
    :param file_name: path for video
    :param mp3_name: path for audio

    :return:
    """

    # replace video audio 
    # subprocess.call('ffmpeg -y -i ' + video_file 
    #                 + ' -i ' + audio_file + ' -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 '
    #                 + outfile_name, shell=True)
    # no orginal audio in the video 
    subprocess.call('ffmpeg -y -i ' + video_file 
                    + ' -i ' + audio_file + ' -c:v copy -c:a aac -strict experimental '
                    + outfile_name, shell=True)


def video_concat(first_video_path, second_video_path, outfile_name='current_add.mp4'):
    with open("./join.txt", "w+") as file:
        file.write("file \'" + first_video_path + "\'")
        file.write("\n")
        file.write("file \'" + second_video_path + "\'")
    
    # concat video
    subprocess.call('ffmpeg -y -f concat -safe 0 -i join.txt -c copy -y '+ outfile_name, shell=True)


def merge_audio_silence(audio_name):
    audio_path = "./alphabet_sounds/" + audio_name + ".mp3"
    # load two mp3
    input_music_1 = AudioSegment.from_mp3(audio_path) 
    input_music_2 = AudioSegment.from_mp3("./alphabet_sounds/silence.mp3") 
    # the DBs
    # input_music_1_db = input_music_1.dBFS
    # input_music_2_db = input_music_2.dBFS
    # the legnths
    # input_music_1_time = len(input_music_1)
    # input_music_2_time = len(input_music_2)
    # adjust DB same
    # db = input_music_1_db - input_music_2_db
    # if db > 0:
    #     input_music_1 += abs(db)
    # elif db < 0:
    #     input_music_2 += abs(db)
    # merge audio
    output_music = input_music_2 + input_music_2 + input_music_1 + input_music_2 + input_music_2 + input_music_2  + input_music_2
    # only need 6000 mil second
    output_music = output_music[:6001] 
    # simple way
    output_path = "./alphabet_sounds/add_silence/" + audio_name + ".mp4"
    output_music.export(output_path , format="mp4")
    # bitrate：byte rate，album：album name，artist，cover：cover image
    # output_music.export("E:/output_music.mp3", format="mp3", bitrate="192k", tags={"album": "test", "artist": "test"}, cover="E:/test.jpg") 


if __name__ == "__main__":
    audio_name_list = ["A", "B", "C", "D", "E", "F", "G"
                        , "H", "I", "J", "K", "L", "M", "N"
                        , "O", "P", "Q", "R", "S", "T", "U"
                        , "V", "W", "X", "Y", "Z"]
    for item in audio_name_list:
        merge_audio_silence(item)
    # video_add_audio("./current.mp4", "./alphabet_sounds/add_silence/I.mp4", "test.mp4")
    # video_add_audio("./alphabet_sounds/test_videos/29.mp4", "./alphabet_sounds/add_silence/B.mp4")
    # video_concat("./current_add_audio.mp4", "./current_add_audio_2.mp4")
