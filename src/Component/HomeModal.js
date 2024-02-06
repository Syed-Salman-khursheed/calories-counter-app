import axios from 'axios';
import Modal from 'react-native-modal';
import Feather from 'react-native-vector-icons/Feather';
import Fontisto from 'react-native-vector-icons/Fontisto';
import storage from '@react-native-firebase/storage';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {
  StyleSheet,
  Dimensions,
  View,
  Text,
  TouchableOpacity,
  Image,
  PermissionsAndroid,
} from 'react-native';
import {ActivityIndicator} from 'react-native-paper';
import {Button, Dialog, Portal} from 'react-native-paper';
import React, {useContext, useState} from 'react';
import {AuthAction} from '../Context/AuthContext';

const windowWidth = Dimensions.get('window').width;

export const HomeModal = ({open, setOpen}) => {
  const {addCalories} = useContext(AuthAction);
  const [photo, setPhoto] = useState();
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [loading, setLoading] = useState(false);
  const [open1, setOpen1] = useState(false);
  const [visible, setVisible] = useState(false);
  const showDialog = () => setVisible(true);
  const hideDialog = () => {
    setVisible(false);
  };

  const options = {
    saveToPhotos: true,
    mediaType: 'photo',
  };
  const openCamera = async () => {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.CAMERA,
    );
    console.log('granted---->', granted);
    // const grantedExternal = await PermissionsAndroid.request(
    //   PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
    //   {
    //     title: 'External Storage Write Permission',
    //     message: 'App needs write permission',
    //   },
    //   );
    // console.log("grantedExternal---->",grantedExternal)
    console.log(
      'PermissionsAndroid.RESULTS.GRANTED---->',
      PermissionsAndroid.RESULTS.GRANTED,
    );
    if (granted === PermissionsAndroid.RESULTS.GRANTED) {
      // && grantedExternal === PermissionsAndroid.RESULTS.GRANTED ){
      const result = await launchCamera(options);
      console.log(granted, result);
      setPhoto(result.assets[0]);
      // uploadImage()
      // setOpen(false)
    }
  };

  const openGallery = async () => {
    const result = await launchImageLibrary(options);
    setPhoto(result.assets[0]);
    // setOpen(false)
  };

  const uploadImage = async () => {
    console.log('chala--->>>>');
    setLoading(true);

    const reference = storage().ref(photo.fileName);
    const pathToFile = photo.uri;

    await reference.putFile(pathToFile);
    const task = reference.putFile(pathToFile);

    console.log('task------>>>>', task);

    task.on('state_changed', taskSnapshot => {
      console.log(
        `${taskSnapshot.bytesTransferred} transferred out of ${taskSnapshot.totalBytes}`,
      );
    });

    task
      .then(async () => {
        console.log('Image uploaded to the bucket!');
        // Get the download URL of the uploaded image
        const url = await reference.getDownloadURL();
        let data = JSON.stringify({
          image: url,
        });

        let config = {
          method: 'post',
          maxBodyLength: Infinity,
          url: 'https://d29c-2603-300a-21e3-0-d131-6e3f-46f3-578f.ngrok-free.app/api/model',
          headers: {
            'Content-Type': 'application/json',
          },
          data: data,
        };

        axios
          .request(config)
          .then(response => {
            console.log(response.data?.data,'response---->>>>')
            const {itemName,calorie,weight}=response.data?.data
            const newData = {
              itemName,
              calorie,
              weight,
              image: url,
            };
            setOpen1(true);
            setData(newData);
            console.log(data,'data---->>>>')

            setLoading(false);
          })
          .catch(error => {
            setLoading(false);
            setError(error);
            showDialog();
            console.log('error--------->>>',error);
          });
      })
      .catch(error => {
        setLoading(false);
        setError(error);
        showDialog();
        console.error('Error uploading image:', error);
      });
  };
  
  const handleSubmit=()=>{
    setOpen1(false)
addCalories(data)
  }
  return (
    <>
      <Modal
        isVisible={open}
        onBackdropPress={() => {
          setOpen(false);
        }}
        //  onSwipeComplete={() => setOpen(false)}
        //  swipeDirection="top"
        style={styles.modelStyle}>
        <View style={styles.modelContainer}>
          {photo?.uri ? (
            <View style={{alignItems: 'center', justifyContent: 'center'}}>
              <Image
                source={{
                  uri: photo?.uri,
                }}
                style={{width: 100, height: 100, borderRadius: 8}}
              />
              <TouchableOpacity
                style={styles.modalbtn}
                onPress={() => {
                  uploadImage();
                }}>
                <Text style={styles.modalbtnText}>
                  {loading ? (
                    <ActivityIndicator animating={true} color="#fff" />
                  ) : (
                    <>
                      Find Calories{' '}
                      <Feather name="camera" color="#fff" size={20} />{' '}
                    </>
                  )}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => {
                  setPhoto('');
                }}
                style={styles.modalbtn}>
                <Text style={styles.modalbtnText}>
                  Back To Snap
                  <Fontisto name="photograph" color="#fff" size={20} />
                </Text>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              <TouchableOpacity onPress={openCamera} style={styles.modalbtn}>
                <Text style={styles.modalbtnText}>
                  Launch Camera <Feather name="camera" color="#fff" size={20} />
                </Text>
              </TouchableOpacity>

              <TouchableOpacity onPress={openGallery} style={styles.modalbtn}>
                <Text style={styles.modalbtnText}>
                  Launch Gallery{' '}
                  <Fontisto name="photograph" color="#fff" size={20} />
                </Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </Modal>
      <Modal
        isVisible={open1}
        onBackdropPress={() => {
          setOpen(false);
        }}
        style={styles.modelStyle1}>
        <View style={styles.modelContainer1}>
          <View style={{alignItems: 'center'}}>
            <Image
              source={{
                uri: data?.image,
              }}
              style={{width: 100, height: 100, borderRadius: 8}}
            />
          </View>
          <View style={{marginLeft: 30, marginRight: 30}}>
            <View
              style={{flexDirection: 'row', justifyContent: 'space-between'}}>
              <Text style={styles.itemKey}>Name</Text>
              <Text style={styles.itemname}>{data?.itemName}</Text>
            </View>
            <View
              style={{flexDirection: 'row', justifyContent: 'space-between'}}>
              <Text style={styles.itemKey}>Calories</Text>
              <Text style={styles.itemname}>{data?.calorie}</Text>
            </View>
            <View
              style={{flexDirection: 'row', justifyContent: 'space-between'}}>
              <Text style={styles.itemKey}>Quantity</Text>
              <Text style={styles.itemname}>{data?.weight}</Text>
            </View>
          </View>

          <View
            style={{
              flexDirection: 'row',
              justifyContent: 'space-around',
              marginBottom: 20,
            }}>
            <TouchableOpacity
              style={styles.cancelbtn}
              onPress={() => {
                setOpen1(false);
              }}>
              <Text style={styles.cancelbtnText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.addbtn} onPress={()=>handleSubmit()}>
              <Text style={styles.addbtnText}>Add To Diet</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Dialog
        visible={visible}
        onDismiss={hideDialog}
        style={{backgroundColor: 'white', color: '#555'}}>
        <Dialog.Title>Something went wrong</Dialog.Title>
    
        <Dialog.Actions>
          <Button onPress={hideDialog} style={{color: '#01714b'}}>
            Done
          </Button>
        </Dialog.Actions>
      </Dialog>
    </>
  );
};

const styles = StyleSheet.create({
  modelStyle: {
    justifyContent: 'flex-end',
    margin: 0,
  },
  modelContainer: {
    backgroundColor: '#fff',
    height: 270,
    borderTopRightRadius: 30,
    borderTopLeftRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalbtn: {
    width: windowWidth * 0.8,
    padding: 15,
    backgroundColor: '#01714b',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    marginTop: 10,
  },
  modalbtnText: {
    textAlign: 'center',
    fontSize: 15,
    fontWeight: 'bold',
    color: '#fff',
  },
  modelStyle1: {
    justifyContent: 'center',
    margin: 30,
  },
  modelContainer1: {
    backgroundColor: '#fff',
    height: 290,
    borderRadius: 20,
    justifyContent: 'space-around',
    // alignItems:'center'
  },
  addbtn: {
    paddingHorizontal: 20,
    padding: 10,
    marginHorizontal: 10,
    backgroundColor: '#01714b',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    marginTop: 10,
    elevation: 5,
  },
  addbtnText: {
    textAlign: 'center',
    fontSize: 15,
    fontWeight: 'bold',
    color: '#fff',
  },
  cancelbtn: {
    padding: 10,
    paddingHorizontal: 20,
    marginHorizontal: 10,
    backgroundColor: '#e6f1ed',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    marginTop: 10,
    elevation: 3,
  },
  cancelbtnText: {
    textAlign: 'center',
    fontSize: 15,
    fontWeight: 'bold',
    color: '#01714b',
  },
  itemKey: {
    marginTop: 1,
    color: 'grey',
    fontWeight: 'bold',
    paddingBottom: 10,
  },
  itemname: {
    marginTop: 1,
    color: '#01714b',
    fontWeight: 'bold',
  },
});
