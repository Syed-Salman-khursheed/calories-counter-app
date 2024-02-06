import * as React from "react";
import { View, Text, Image, StyleSheet, TouchableOpacity } from "react-native";
// import Skeleton from "@mui/material/Skeleton";
import MaterialIcons from 'react-native-vector-icons/MaterialIcons';

export default function Foodtime({ name, img, navigation,handleOpen }) {

  return (
    <TouchableOpacity onPress={()=>{handleOpen()}}  activeOpacity={0.5}>
      <View style={styles.container}>
        <View style={styles.body}>
          <Image
            style={styles.img}
            source={{ uri: img }}
          />
          <Text style={styles.name}>{name}</Text>
          <MaterialIcons name="add" size={35} style={styles.addIcon} color="#fff" />
        </View>
      </View>
    </TouchableOpacity>
  );
}
const styles = StyleSheet.create({

  body: {
    padding: 8,
    backgroundColor: "#01714b",
    cursor: "pointer",
    flexDirection: "row",
    borderRadius: 10,
    justifyContent: "space-between",
    marginHorizontal:15,
    marginVertical:5,
    height: 75,
    elevation:2,
  },
  img: {
    borderRadius: 3,
    width: 60,
    height: 60,

  },
  name: {
    padding: 1,
    fontSize: 18,
    alignSelf: 'center',
    color: '#fff',
    // paddingTop:10,
    fontWeight: 'bold',
  },
  addIcon: {
    alignSelf: 'center',
    justifyContent: 'center'
  }
});
