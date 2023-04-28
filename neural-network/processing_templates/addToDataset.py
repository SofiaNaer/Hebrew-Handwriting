import os
import uuid
import shutil
from split_image import Split
class addToDataset:
    def __init__(self):
        self.input_dir = "filled_in_templates/result"
        self.output = '../../dataset'



        self.add()
        self.delete_folder(self.input_dir)


    def add (self):
        binary_dir = '00000'
        list_alef = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
        list_bet = ['10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg', '17.jpg', '18.jpg']
        list_gimel = ['19.jpg', '20.jpg', '21.jpg', '22.jpg', '23.jpg', '24.jpg', '25.jpg', '26.jpg', '27.jpg']
        list_dalet = ['28.jpg', '29.jpg', '30.jpg', '31.jpg', '32.jpg', '33.jpg', '34.jpg', '35.jpg', '36.jpg']
        list_hey = ['37.jpg', '38.jpg', '39.jpg', '40.jpg', '41.jpg', '42.jpg', '43.jpg', '44.jpg', '45.jpg']
        list_vav = ['46.jpg', '47.jpg', '48.jpg', '49.jpg', '50.jpg', '51.jpg', '52.jpg', '53.jpg', '54.jpg']
        list_z = ['55.jpg', '56.jpg', '57.jpg', '58.jpg', '59.jpg', '60.jpg', '61.jpg', '62.jpg', '63.jpg']
        list_het = ['64.jpg', '65.jpg', '66.jpg', '67.jpg', '68.jpg', '69.jpg', '70.jpg', '71.jpg', '72.jpg']
        list_tet = ['73.jpg', '74.jpg', '75.jpg', '76.jpg', '77.jpg', '78.jpg', '79.jpg', '80.jpg', '81.jpg']
        list_iud = ['82.jpg', '83.jpg', '84.jpg', '85.jpg', '86.jpg', '87.jpg', '88.jpg', '89.jpg', '90.jpg']
        list_haf = ['91.jpg', '92.jpg', '93.jpg', '94.jpg', '95.jpg', '96.jpg', '97.jpg', '98.jpg', '99.jpg']
        list_hafs = ['100.jpg', '101.jpg', '102.jpg', '103.jpg', '104.jpg', '105.jpg', '106.jpg', '107.jpg', '108.jpg']
        list_lamed = ['109.jpg', '110.jpg', '111.jpg', '112.jpg', '113.jpg', '114.jpg', '115.jpg', '116.jpg', '117.jpg']
        list_mem = ['118.jpg', '119.jpg', '120.jpg', '121.jpg', '122.jpg', '123.jpg', '124.jpg', '125.jpg', '126.jpg']
        list_mems = ['127.jpg', '128.jpg', '129.jpg', '130.jpg', '131.jpg', '132.jpg', '133.jpg', '134.jpg', '135.jpg']
        list_nun = ['136.jpg', '137.jpg', '138.jpg', '139.jpg', '140.jpg', '141.jpg', '142.jpg', '143.jpg', '144.jpg']
        list_nuns = ['145.jpg', '146.jpg', '147.jpg', '148.jpg', '149.jpg', '150.jpg', '151.jpg', '152.jpg', '153.jpg']
        list_sameh = ['154.jpg', '155.jpg', '156.jpg', '157.jpg', '158.jpg', '159.jpg', '160.jpg', '161.jpg', '162.jpg']
        list_ain = ['163.jpg', '164.jpg', '165.jpg', '166.jpg', '167.jpg', '168.jpg', '169.jpg', '170.jpg', '171.jpg']
        list_pei = ['172.jpg', '173.jpg', '174.jpg', '175.jpg', '176.jpg', '177.jpg', '178.jpg', '179.jpg', '180.jpg']
        list_peis = ['181.jpg', '182.jpg', '183.jpg', '184.jpg', '185.jpg', '186.jpg', '187.jpg', '188.jpg', '189.jpg']
        list_zadik = ['190.jpg', '191.jpg', '192.jpg', '193.jpg', '194.jpg', '195.jpg', '196.jpg', '197.jpg', '198.jpg']
        list_zadiks = ['199.jpg', '200.jpg', '201.jpg', '202.jpg', '203.jpg', '204.jpg', '205.jpg', '206.jpg',
                       '207.jpg']
        list_kuf = ['208.jpg', '209.jpg', '210.jpg', '211.jpg', '212.jpg', '213.jpg', '214.jpg', '215.jpg', '216.jpg']
        list_raish = ['217.jpg', '218.jpg', '219.jpg', '220.jpg', '221.jpg', '222.jpg', '223.jpg', '224.jpg', '225.jpg']
        list_shin = ['226.jpg', '227.jpg', '228.jpg', '229.jpg', '230.jpg', '231.jpg', '232.jpg', '233.jpg', '234.jpg']
        list_tav = ['235.jpg', '236.jpg', '237.jpg', '238.jpg', '239.jpg', '240.jpg', '241.jpg', '242.jpg', '243.jpg']

        for root, dirs, files in os.walk(self.input_dir):
            for file_name in files:
                if file_name in list_alef:
                    binary_dir = '00000'
                elif file_name in list_bet:
                    binary_dir = '00001'
                elif file_name in list_gimel:
                    binary_dir = '00010'
                elif file_name in list_dalet:
                    binary_dir = '00011'
                elif file_name in list_hey:
                    binary_dir = '00100'
                elif file_name in list_vav:
                    binary_dir = '00101'
                elif file_name in list_z:
                    binary_dir = '00110'
                elif file_name in list_het:
                    binary_dir = '00111'
                elif file_name in list_tet:
                    binary_dir = '01000'
                elif file_name in list_iud:
                    binary_dir = '01001'
                elif file_name in list_haf:
                    binary_dir = '01010'
                elif file_name in list_hafs:
                    binary_dir = '01011'
                elif file_name in list_lamed:
                    binary_dir = '01100'
                elif file_name in list_mem:
                    binary_dir = '01101'
                elif file_name in list_mems:
                    binary_dir = '01110'
                elif file_name in list_nun:
                    binary_dir = '01111'
                elif file_name in list_nuns:
                    binary_dir = '10000'
                elif file_name in list_sameh:
                    binary_dir = '10001'
                elif file_name in list_ain:
                    binary_dir = '10010'
                elif file_name in list_pei:
                    binary_dir = '10011'
                elif file_name in list_peis:
                    binary_dir = '10100'
                elif file_name in list_zadik:
                    binary_dir = '10101'
                elif file_name in list_zadiks:
                    binary_dir = '10110'
                elif file_name in list_kuf:
                    binary_dir = '10111'
                elif file_name in list_raish:
                    binary_dir = '11000'
                elif file_name in list_shin:
                    binary_dir = '11001'
                elif file_name in list_tav:
                    binary_dir = '11010'




                old_path = f"{root}/{file_name}"
                unique_id = uuid.uuid4()
                f_name = str(unique_id)
                new_path = f"{root}/{f_name}.jpg"
                os.rename(old_path, new_path)
                dst_dir = f"{self.output}/{binary_dir}"
                shutil.move(new_path, dst_dir)





    def delete_folder (self, path):
        try:
            # remove the folder and all its contents
            shutil.rmtree(path)
            print("Folder deleted successfully")
        except OSError as error:
            print(f"Error: {path} : {error.strerror}")



add = addToDataset()


