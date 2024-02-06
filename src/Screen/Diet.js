import { View,Text,  StyleSheet, ScrollView } from 'react-native'
import Icon from 'react-native-vector-icons/Ionicons';
import HomeHeader from '../Component/HomeHeader';
import DietDish from '../Component/DietDish';
import React, {useContext,useEffect} from 'react';
import {AuthAction,AuthContext} from '../Context/AuthContext';

export default function DietHistory({ navigation }) {

  const {getCalories} = useContext(AuthAction);

  const {calories,loading} = useContext(AuthContext);
console.log(calories,'in diet')

 useEffect(()=>{
   getCalories()
 },[])


  return (
    <ScrollView shhowsVerticalScrollIndicator={false}>
      <View style={styles.container}>
        <HomeHeader
          navigation={navigation}
          name='Diet History'
          icon={<Icon name="arrow-back" size={26} color="#fff" />}
        />
        {loading ? (
          <Text style={{ color: '#555',justifyContent:'center',alignItems:'center' }}>Loading...</Text>
        ) : (
          <>
            {calories?.map((e, i) => (
              <DietDish {...e} key={i} />
            ))}
          </>
        )}
       
      </View>
    </ScrollView>

  )
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f9f9f9f9",
  },

});
