import { StyleSheet, Text, View, TouchableOpacity, Alert, Platform } from 'react-native';
import ParallaxScrollView from '@/components/ParallaxScrollView'; // Ensure this is correct
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { useState } from 'react';

// Define TeamMember type to avoid implicit typing errors
type TeamMember = {
  name: string;
  regNo: string;
  dept: string;
};

export default function TabTwoScreen() {
  // Team member details with explicit types
  const teamMembers: TeamMember[] = [
    { name: 'Arun G', regNo: 'RA2432014020042', dept: 'M.Sc.Applied Data Science' },
    { name: 'Soniya Agarwal', regNo: 'RA2432014020043', dept: 'M.Sc.Applied Data Science' },
    { name: 'Rahul P', regNo: 'RA2432014020044', dept: 'M.Sc.Applied Data Science' },
    { name: 'Jai Adithya Ram Kumar P', regNo: 'RA2432014020045', dept: 'M.Sc.Applied Data Science' },
    { name: 'Aswin S', regNo: 'RA2432014020046', dept: 'M.Sc.Applied Data Science' },
  ];

  // Manage the visibility of the dropdown menu for each team member
  const [expandedMember, setExpandedMember] = useState<string | null>(null);

  // Handle member click event to toggle dropdown visibility
  const handleMemberClick = (name: string) => {
    setExpandedMember(expandedMember === name ? null : name); // Toggle visibility
  };

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#D0D0D0', dark: '#353636' }}
      headerComponent={
        <IconSymbol
          size={310}
          color="#808080"
          name="chevron.left.forwardslash.chevron.right"
          style={styles.headerImage}
        />
      }
    >
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Idea Generator</ThemedText>
      </ThemedView>
      <ThemedText>This app generates ideas for projects based on your input.</ThemedText>

      <ThemedView style={styles.teamContainer}>
        <ThemedText type="title">Team Members</ThemedText>
        {teamMembers.map((member, index) => (
          <View key={index}>
            <TouchableOpacity
              onPress={() => handleMemberClick(member.name)}
              style={styles.memberButton}
            >
              <ThemedText>{member.name}</ThemedText>
            </TouchableOpacity>
            {expandedMember === member.name && (
              <View style={styles.dropdownMenu}>
                <ThemedText>Registration No: {member.regNo}</ThemedText>
                <ThemedText>Department: {member.dept}</ThemedText>
              </View>
            )}
          </View>
        ))}
      </ThemedView>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  headerImage: {
    color: '#808080',
    bottom: -90,
    left: -35,
    position: 'absolute',
  },
  titleContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  teamContainer: {
    marginTop: 20,
  },
  memberButton: {
    padding: 10,
    marginVertical: 5,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    alignItems: 'center',
  },
  dropdownMenu: {
    padding: 10,
    backgroundColor: '#e0e0e0',
    borderRadius: 8,
    marginTop: 5,
  },
});
