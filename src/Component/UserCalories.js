//1182540-0336
import Icon from 'react-native-vector-icons/Ionicons';
import { StyleSheet,Dimensions,View,Text,  } from "react-native";
import React, {useContext, useEffect, useState} from 'react';
import {AuthAction, AuthContext} from '../Context/AuthContext';
import AsyncStorage from '@react-native-async-storage/async-storage';
const windowWidth = Dimensions.get('window').width;

export const UserCalories = () => {

  const {userInfo} = useContext(AuthContext);
  const {getTodayCalories} = useContext(AuthAction);

  console.log(userInfo,'setuserInfo state in context------->>>>in calories page>')



  return (
    <View style={styles.calories}>
    <Icon name="stats-chart-outline" style={styles.menuIcon} size={80} color="#01714b" />
    <View style={styles.container}>
    <View style={styles.caloriesInfo}>
      <Text style={styles.caloriesHeading}>Calories Required</Text>
      <Text style={styles.caloriesQty}>{userInfo?.perDayCalories}</Text>
    </View>
    <View style={styles.caloriesInfo}>
      <Text style={styles.caloriesHeading}>Calories Consume</Text>
      <Text style={styles.caloriesQty}>1150</Text>
    </View>
    <View style={styles.caloriesInfo}>
      <Text style={styles.caloriesHeading}>Calories Remaining</Text>
      <Text style={styles.caloriesQty}>1350</Text>
    </View>
    </View>
  </View>

  )
}

const styles=StyleSheet.create({
    calories: {
        // paddingVertical: 1,
        paddingHorizonta:15,
        flexDirection: 'row',
        justifyContent: 'space-between',
        backgroundColor: "white",
        cursor: "pointer",
        margin: 10,
        marginHorizontal: 15,
        borderRadius: 3,
        elevation: 5,
        
      },
      container:{
        justifyContent: "center",

      },
      caloriesInfo: {
        padding: 5,
        flexDirection: "row",
        alignItems:'space-between',
        color:'#01714b'
      },
      caloriesHeading:{
        color:'#01714b'

      },
      caloriesQty: {
        paddingLeft: 20,
        fontWeight: 'bold',
        color:'#01714b',
        alignSelf:'flex-end'
      },
})
