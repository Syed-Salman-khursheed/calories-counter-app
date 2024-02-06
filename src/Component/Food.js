import React from "react";
import { StyleSheet,  View } from "react-native";
import Foodtime from "./Foodtime";

export default function Foods({handleOpen,navigation}){
    return(
        <View style={styles.timecontainer}>
         
            <Foodtime
            handleOpen={handleOpen}
              navigation={navigation}
              name="Break Fast"
              img="https://images.pexels.com/photos/103124/pexels-photo-103124.jpeg?auto=compress&cs=tinysrgb&w=600"
            />
            <Foodtime
            handleOpen={handleOpen}
            navigation={navigation}
              name="Lunch"
              img="https://static.toiimg.com/photo/90934370.cms"
            />
            <Foodtime
            handleOpen={handleOpen}
            navigation={navigation}
              name="Dinner"
              img="https://cdn.vox-cdn.com/thumbor/bMDwxq8pYW4fuLHBGhcVKTl9PX8=/0x0:5700x3800/1200x800/filters:focal(2394x1444:3306x2356)/cdn.vox-cdn.com/uploads/chorus_image/image/65460644/GettyImages_1032865588.0.jpg"
            />
            {/* <Foodtime
              navigation={navigation}
              name="Water"
              img="https://media.gettyimages.com/photos/drinking-glass-of-water-picture-id484782395?s=612x612"
            /> */}

          </View>
    )
}
const styles = StyleSheet.create({
   
    timecontainer: {
      marginTop: 10,
    }
})