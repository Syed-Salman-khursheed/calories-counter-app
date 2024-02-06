import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image ,PermissionsAndroid} from 'react-native';
import {launchCamera,launchImageLibrary} from 'react-native-image-picker';

export default function CameraScreen ({navigation}){

  const [photo,setPhoto]=useState()
  const [photo1,setPhoto1]=useState()

  const options={
    saveToPhotos:true,
    mediaType:'photo'
  }
 const openCamera =  async()=>{
  const granted= await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA)
  console.log("granted---->",granted)
  // const grantedExternal = await PermissionsAndroid.request(
  //   PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
  //   {
  //     title: 'External Storage Write Permission',
  //     message: 'App needs write permission',
  //   },
  //   );
    // console.log("grantedExternal---->",grantedExternal)
    console.log("PermissionsAndroid.RESULTS.GRANTED---->",PermissionsAndroid.RESULTS.GRANTED)
  if(granted === PermissionsAndroid.RESULTS.GRANTED ){
    // && grantedExternal === PermissionsAndroid.RESULTS.GRANTED ){
    const result = await launchCamera(options)
      console.log(granted, result)
    setPhoto(result.assets[0].uri)
  } 
}

const openGallery=async()=>{
  const result=await launchImageLibrary(options)
  setPhoto1(result.assets[0].uri)
}

  return (
    <View style={styles.container}>
       <Image
        source={{
          uri:photo
        }}
        style={{ width: 100, height: 100 }}
      />
    
      <TouchableOpacity onPress={openCamera} style={styles.button}>
        <Text style={styles.buttonText}>Launch Camera Directly</Text>
      </TouchableOpacity>
      <Image
        source={{
          uri:photo1
        }}
        style={{ width: 100, height: 100 }}
      />
      <TouchableOpacity onPress={openGallery} style={styles.button}>
        <Text style={styles.buttonText}>Launch Galler Directly</Text>
      </TouchableOpacity>

      <TouchableOpacity onPress={() => navigation.navigate("Home")} style={styles.button}>
        <Text style={styles.buttonText}>Back to Home</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 30,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
  button: {
    width: 250,
    height: 60,
    backgroundColor: '#3740ff',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 4,
    marginBottom: 12,
  },
  buttonText: {
    textAlign: 'center',
    fontSize: 15,
    color: '#fff',
  },
});


