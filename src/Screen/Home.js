import Icon from 'react-native-vector-icons/Ionicons';
import { StyleSheet, Text, View, SafeAreaView, ScrollView, StatusBar,TouchableOpacity ,Image, Dimensions } from "react-native";
import HomeHeader from "../Component/HomeHeader";
import Banner from "../Component/Banner";
import Foods from "../Component/Food";
import Dishes from "../Component/Dishes";
import { UserCalories } from "../Component/UserCalories";
import { HomeModal } from "../Component/HomeModal";
import AsyncStorage from "@react-native-async-storage/async-storage";
import React, {useContext,useState,useEffect} from 'react';
import {AuthAction,AuthContext} from '../Context/AuthContext';
const windowWidth = Dimensions.get('window').width;

export default function Home({ navigation }) {
  const [open,setOpen]=useState(false)
  const {getUserInfo,getCalories} = useContext(AuthAction);
  const {user} = useContext(AuthContext);
console.log('user stae in context-------------->>>',user)

  const handleOpen=()=>{
    setOpen(true)
  }
  
  useEffect(() => {
    registerSplash();
    
  },[]);

  const registerSplash = async () => {
    const getUser = await AsyncStorage.getItem('User');


      if (getUser === null) {
        navigation.navigate('Login');
      }
  };

  return (
    <ScrollView>
      <View style={styles.container}>
        <StatusBar animated={true} style={'dark'} />
        <SafeAreaView>
          <HomeHeader navigation={navigation} name='Home' icon={<Icon name="menu-sharp" style={styles.menuIcon} size={26} color="#fff"/>} />
           <UserCalories/>
          
          <Banner />
          <Foods handleOpen={handleOpen} navigation={navigation}/>
          <Dishes/>
          

        </SafeAreaView>
      </View>
    <HomeModal open={open} setOpen={setOpen}/>
    </ScrollView>
    
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: windowWidth,
    backgroundColor: '#ffffff',
    // backgroundColor:'#FEFCFF',
  },
});
