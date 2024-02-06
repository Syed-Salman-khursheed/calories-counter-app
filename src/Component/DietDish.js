import React from 'react'
import { View, Text,StyleSheet,Image } from 'react-native';
export default function DietDish({name,image,calories,qty,createdAt}){
  return (
    <View style={styles.body}>
  <View style={{flexDirection:'row'}}>
  <Image
   style={styles.img}
   source={{ uri: image }}
 />
  
   <View style={styles.dishInfo}>
   <Text style={styles.dishname}>{name}</Text>
   <Text style={styles.quantity}>{qty}</Text>
   </View>
  </View>
   <View style={{justifyContent:'space-between'}}>
   <Text style={styles.dishcaloires}>{calories}</Text>
   <Text style={styles.dishtime}>{createdAt}</Text>
   </View>
     </View>    
  );
}
const styles = StyleSheet.create({
    body: {

        paddingVertical: 10,
        paddingHorizontal: 10,
        backgroundColor: "#ffffff",
        flexDirection: "row",
        justifyContent:'space-between',
        borderRadius: 6,
        marginHorizontal:6,
        marginTop:5,
        marginBottom:2,
        elevation: 2,
      },
      img: {
        borderRadius: 3,
        width: 70,
        height: 70,
        // borderRadius:50,
      },
      dishname: {
        fontSize: 16,
        color: '#555',
        fontWeight: 'bold',
      },
      dishcaloires: {
        alignSelf:'flex-end',
        color: '#01714b',
        fontWeight: 'bold',
        fontSize: 18,
    },
      dishtime: {
        alignSelf:'flex-end',
        fontSize: 13,
        color: 'grey'
    },
      quantity: {
        fontSize: 15,
        color: 'grey'
    },
      dishInfo:{
        marginLeft:10,
      },
      message: {
        fontSize: 13, 
        color: '#555',
      },
})
