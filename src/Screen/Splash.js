import {
    StyleSheet,
    Text,
    View,
    Image
    ,Dimensions
  } from "react-native";
  import * as yup from 'yup'
  import { Formik } from 'formik'
  import { InputField } from "./Register";
  import image from './kcal.png'
  import React, {useContext, useState, useEffect} from 'react';
  import AsyncStorage from "@react-native-async-storage/async-storage";
 const windowWidth = Dimensions.get('window').width;

  
  export default function Splash({ navigation }) {
  

  
    useEffect(() => {
       registerSplash();
    });
  
    const registerSplash = async () => {
      const getUser = await AsyncStorage.getItem('User');
        if (getUser !== null) {
          navigation.navigate('Home');
        }
        else{
            navigation.navigate('Login')
        }
    };
  
    return (
      <View style={styles.container}>
      <Image
      style={styles.img}
      source={image}
    />
          <Text style={styles.heading}>Stay Healthy</Text>
          <Text style={styles.heading}>With</Text>
          <Text style={styles.heading}>Calories Counter</Text>
  
        
      </View>
    );
  }
  const styles = StyleSheet.create({

    container: {
      width:windowWidth,
      flex: 1,
      backgroundColor: '#01714b',
      justifyContent:'center',
      alignItems:'center',
    //   marginHorizontal:50,
    },
    heading: {
      color: "#fafafa",
      fontSize: 36,
      fontWeight:'bold',
      fontStyle:'italic'

    },
    img:{
        // height:300,
        widtht:200,
    }


  
  });
  