import { View, Text,TextInput, StyleSheet, ScrollView ,TouchableOpacity,Dimensions} from 'react-native'
import Icon from 'react-native-vector-icons/Ionicons';
import Feather from "react-native-vector-icons/Feather";
import Modal from "react-native-modal";
import HomeHeader from '../Component/HomeHeader';
import Comment from '../Component/Comment';
import React, {useContext, useState,useEffect} from 'react';
import {AuthAction,AuthContext} from '../Context/AuthContext';
const windowWidth = Dimensions.get('window').width;


export default function Feedback({ navigation }) {

  const [open, setOpen] = useState(false)
  const [message, setMessage] = useState('')
  const {addFeedback,getFeedback} = useContext(AuthAction);
  const {feedbacks} = useContext(AuthContext);

  const handleInputChange = (text) => {
    setMessage(text);
  };
 
  const handleSubmit=()=>{
    if(message){
      setOpen(false)
      addFeedback(message)
       setMessage('')
    }
    else{
      alert('plz enter feedback first')
    }
  }

  useEffect(()=>{
    getFeedback()
  },[])

  return (
    <ScrollView shhowsVerticalScrollIndicator={false}>
      <View style={styles.container}>
        <HomeHeader
          navigation={navigation}
          name='Feedback'
          icon={<Icon name="arrow-back" size={26} color="#fff" />}
          editIcon={
            <TouchableOpacity onPress={()=>setOpen(true)}>
              <Feather name="edit" size={22} color="#fff" />
            </TouchableOpacity>}
        />
    
    
      {feedbacks?.map((e,i)=><Comment {...e} key={i}/>)}
        
      </View>
      <Modal isVisible={open} 
      onBackdropPress={() => {
        setOpen(false);
      }} 
      style={styles.modelStyle} 
      >
        <View style={styles.modelContainer}>

        <TextInput
        value={message}
        onChangeText={handleInputChange}
        style={styles.input}
        multiline={true}
        numberOfLines={5}
        placeholder='enter your feedback here'
      />
          <TouchableOpacity style={styles.addbtn} onPress={()=>{handleSubmit()}}>
            <Text style={styles.addbtnText}>Submit</Text>
          </TouchableOpacity>

        </View>
      </Modal>
    </ScrollView>

  )
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f9f9f9f9",
  },
  modelStyle: {
    justifyContent: 'center',
    margin: 30,
  },
  modelContainer: {
    backgroundColor: '#fff',
    paddingHorizontal:25,
    paddingVertical:30,
    borderRadius: 20,
    justifyContent: 'space-around',
  },
  input: {
    paddingLeft:10,
    borderRadius:10,
    fontSize:18,
    backgroundColor: '#e6f1ed',
    color:'grey',
    textAlignVertical: 'top'
  },
  addbtn: {
    paddingHorizontal: 20,
    padding: 10,
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
});
